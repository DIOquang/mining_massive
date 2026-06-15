from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import redis
import json
import hashlib
import logging
import uuid
from typing import List
from qdrant_client import QdrantClient
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
PROJECT_ID      = "project-528e2858-1a08-4d22-bcd"
REGION          = "us-central1"
QDRANT_HOST     = "34.41.86.165"
QDRANT_PORT     = 6333
COLLECTION_NAME = "amazon_items"
REDIS_HOST      = "10.69.141.123"  # GCP Memorystore for Redis
REDIS_PORT      = 6379
VECTOR_DIM      = 384
TOP_K_RETRIEVE  = 50
TOP_K_FINAL     = 10

# ─── KHỞI TẠO ────────────────────────────────────────────────────────────────
import os
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"))
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
# NOTE: Metadata sản phẩm được lưu trong Qdrant payload — không cần Firestore

try:
    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=2)
    cache.ping()
    log.info("✅ Redis (Memorystore) kết nối thành công!")
except Exception:
    cache = None
    log.warning("⚠️ Redis không khả dụng, bỏ qua caching.")


app = FastAPI(title="Two-Tower Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCHEMA ──────────────────────────────────────────────────────────────────
class SessionRequest(BaseModel):
    clicked_item_ids: List[str]   # danh sách item_id người dùng đã click


class RecommendItem(BaseModel):
    item_id: str
    title: str
    score: float
    explanation: str
    image: str = ""
    rating: float = 0.0
    review_count: int = 0


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
    session_size: int
    from_cache: bool

# ─── HÀM HỖ TRỢ ──────────────────────────────────────────────────────────────
def get_item_vectors(item_ids: List[str]) -> List[List[float]]:
    """Truy xuất vectors của các item từ Qdrant."""
    import uuid
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, str(iid))) for iid in item_ids]
    results   = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=point_ids,
        with_vectors=True,
    )
    vectors = [r.vector for r in results if r.vector is not None]
    return vectors


def compute_session_vector(vectors: List[List[float]]) -> List[float]:
    """Tính Session Vector bằng Mean Pooling."""
    arr = np.array(vectors, dtype=np.float32)
    mean_vec = arr.mean(axis=0)
    # L2 normalize
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.tolist()


def search_qdrant(session_vector: List[float], top_k: int) -> list:
    """Tìm kiếm ANN trong Qdrant."""
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=session_vector,
        limit=top_k,
        with_payload=True,
    )
    return results


def rerank_with_openai(session_context: str, candidates: list) -> List[dict]:
    """Dùng OpenAI để Rerank Top-10 và sinh giải thích."""
    candidate_text = "\n".join([
        f"{i+1}. ID={c.payload['item_id']}, Tên={c.payload.get('title','?')}, "
        f"Điểm tương đồng={c.score:.3f}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""Bạn là hệ thống gợi ý sản phẩm thông minh trên sàn thương mại điện tử Amazon.

Người dùng mới (chưa có lịch sử) vừa thể hiện sự quan tâm đến các sản phẩm sau trong phiên mua sắm hiện tại:
{session_context}

Dưới đây là danh sách {len(candidates)} sản phẩm ứng viên được hệ thống AI tìm kiếm tương đồng:
{candidate_text}

Hãy chọn ra đúng {TOP_K_FINAL} sản phẩm phù hợp nhất và trả lời theo định dạng JSON sau:
{{
  "recommendations": [
    {{
      "rank": 1,
      "item_id": "...",
      "title": "...",
      "explanation": "Giải thích ngắn gọn tại sao gợi ý sản phẩm này (1-2 câu, bằng tiếng Việt)"
    }}
  ]
}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        return data.get("recommendations", [])
    except Exception as e:
        log.error(f"OpenAI failed: {e}")
        # Fallback: return original candidates with a generic reason
        fallback = []
        for c in candidates[:TOP_K_FINAL]:
            fallback.append({
                "item_id": c.payload["item_id"],
                "title": c.payload.get("title", "Unknown"),
                "explanation": "Gợi ý tự động dựa trên độ tương đồng (Fallback từ OpenAI)."
            })
        return fallback


def build_cache_key(clicked_ids: List[str]) -> str:
    """Tạo cache key từ danh sách item_id."""
    joined = ",".join(sorted(clicked_ids))
    return "rec:" + hashlib.sha256(joined.encode()).hexdigest()


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "qdrant": QDRANT_HOST, "redis": cache is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: SessionRequest):
    if not req.clicked_item_ids:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 item_id để gợi ý.")

    # 1. Kiểm tra cache
    cache_key = build_cache_key(req.clicked_item_ids)
    if cache:
        cached = cache.get(cache_key)
        if cached:
            log.info("✅ Cache HIT!")
            recs = json.loads(cached)
            return RecommendResponse(
                recommendations=[RecommendItem(**r) for r in recs],
                session_size=len(req.clicked_item_ids),
                from_cache=True,
            )

    # 2. Lấy vectors của các item đã click
    vectors = get_item_vectors(req.clicked_item_ids)
    if not vectors:
        raise HTTPException(status_code=404, detail="Không tìm thấy embedding cho các item đã cung cấp.")

    # 3. Tính Session Vector (Mean Pooling)
    session_vec = compute_session_vector(vectors)

    # 4. Tìm kiếm ANN trong Qdrant
    candidates = search_qdrant(session_vec, top_k=TOP_K_RETRIEVE)
    if not candidates:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm liên quan.")

    # 5. Lấy tên các sản phẩm đã click để làm context cho LLM
    clicked_titles = [c.payload.get("title", c.payload["item_id"]) for c in candidates[:3]]
    session_context = ", ".join(clicked_titles)

    # 6. Rerank bằng OpenAI
    reranked = rerank_with_openai(session_context, candidates)

    # 7. Ghép điểm score từ Qdrant và sinh Mock data
    score_map = {c.payload["item_id"]: c.score for c in candidates}
    result = []
    import random
    for r in reranked:
        iid = r["item_id"]
        rng = random.Random(iid)
        # Vì không truyền category vào recommend, lấy tạm "electronics" để có ảnh
        mock_title, mock_image = get_mock_title_and_image("electronics", iid, rng)
        
        result.append(RecommendItem(
            item_id     = iid,
            title       = mock_title,
            score       = score_map.get(iid, 0.0),
            explanation = r["explanation"],
            image       = mock_image,
            rating      = round(rng.uniform(3.5, 5.0), 1),
            review_count= rng.randint(10, 5000)
        ))

    # 8. Lưu cache (TTL 1 giờ)
    if cache:
        cache.setex(cache_key, 3600, json.dumps([r.dict() for r in result]))

    return RecommendResponse(
        recommendations=result,
        session_size=len(req.clicked_item_ids),
        from_cache=False,
    )


def get_mock_title_and_image(category: str, iid: str, rng) -> tuple:
    titles = {
        "electronics": ['Tai nghe Sony WH-1000XM5', 'Chuột Logitech MX Master 3', 'Bàn phím cơ Keychron K2', 'Webcam Logitech C920', 'SSD Samsung 1TB', 'Màn hình LG 27" 4K', 'Laptop MacBook Air M2', 'iPad Pro 12.9"', 'Apple Watch Series 9', 'AirPods Pro 2', 'Loa JBL Charge 5', 'Ổ cứng di động WD 2TB', 'Router WiFi 6 TP-Link', 'Bàn phím HHKB Pro', 'Micro Blue Yeti', 'Card màn hình RTX 4070'],
        "books":       ['Nhà Giả Kim - Paulo Coelho', 'Đắc Nhân Tâm', 'Sapiens - Lược Sử Loài Người', 'The Alchemist', 'Deep Work', 'Atomic Habits', 'Thinking Fast and Slow', 'Rich Dad Poor Dad', '48 Laws of Power', 'The Psychology of Money', 'Zero to One', 'The Lean Startup', 'Clean Code', 'Design Patterns', 'The Pragmatic Programmer', 'Structure and Interpretation'],
        "fashion":     ['Áo polo Ralph Lauren', "Quần jean Levi's 501", 'Áo hoodie Nike Tech Fleece', 'Giày Adidas Ultraboost', 'Túi xách Michael Kors', 'Đồng hồ Seiko', 'Kính mắt Ray-Ban Aviator', 'Áo khoác The North Face', 'Sneakers New Balance 574', 'Áo thun Uniqlo', 'Quần âu Zara', 'Giày Dr. Martens', 'Mũ bucket Columbia', 'Balo Herschel', 'Ví da Coach', 'Thắt lưng Fossil'],
        "home":        ['Nồi cơm điện Zojirushi', 'Máy lọc không khí Xiaomi', 'Robot hút bụi Roomba', 'Nồi chiên không dầu Philips', 'Đèn bàn Xiaomi Mi', 'Máy pha cà phê Nespresso', 'Bộ chăn ga Tencel', 'Giá sách IKEA', 'Đèn ngủ Philips Hue', 'Máy xay sinh tố Vitamix', 'Nồi áp suất Instant Pot', 'Máy lọc nước RO', 'Tủ giày thông minh', 'Gương LED phòng tắm', 'Máy sấy quần áo', 'Lò vi sóng Panasonic'],
        "sports":      ['Giày chạy bộ Nike Pegasus', 'Máy tập đa năng Bowflex', 'Dây kháng lực set 5 cái', 'Thảm yoga TPE 6mm', 'Bình nước Hydro Flask', 'Áo tập Under Armour', 'Bóng rổ Spalding', 'Gậy tennis Wilson', 'Bao tay boxing Everlast', 'Xe đạp tập Peloton', 'Quả tạ điều chỉnh 20kg', 'Băng quấn cổ tay', 'Kính bơi Speedo', 'Áo vest bơi TYR', 'Túi gym Nike', 'Máy đo nhịp tim Garmin'],
    }
    kw = {
        "electronics": "electronics,gadget",
        "books": "book,cover",
        "fashion": "clothing,fashion",
        "home": "furniture,home",
        "sports": "sports,equipment"
    }
    cat = category.lower()
    if cat not in titles:
        cat = "electronics"
        
    cat_titles = titles[cat]
    title = rng.choice(cat_titles)
    
    keyword = kw.get(cat, "product")
    import hashlib
    lock = int(hashlib.md5(iid.encode()).hexdigest(), 16) % 1000 + 1
    image_url = f"https://loremflickr.com/320/240/{keyword}?lock={lock}"
    
    return title, image_url

@app.get("/products")
def get_products(category: str = "electronics", limit: int = 24):
    """Lấy danh sách sản phẩm theo danh mục (tạo vector giả để mô phỏng phân loại)."""
    import random
    import hashlib

    # Tạo vector giả định hướng dựa trên tên danh mục (để các tab ra kết quả khác biệt và ổn định)
    seed = int(hashlib.md5(category.encode()).hexdigest(), 16)
    np.random.seed(seed % (2**32))
    query_vector = np.random.randn(VECTOR_DIM).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = search_qdrant(query_vector.tolist(), top_k=limit)
    
    products = []
    for r in results:
        iid = r.payload.get("item_id", str(r.id))
        rng = random.Random(iid) # Deterministic random based on item_id
        mock_title, mock_image = get_mock_title_and_image(category, iid, rng)
        
        # Chúng ta dùng title gốc (review) làm context, và mock_title làm tên sản phẩm chính
        original_title = r.payload.get("title", "")
        if original_title:
            context = original_title + " | " + r.payload.get("context", "")
        else:
            context = r.payload.get("context", "")

        products.append({
            "item_id": iid,
            "title":   mock_title,
            "context": context,
            "score":   r.score,
            "image":   mock_image,
            "rating":  round(rng.uniform(3.5, 5.0), 1),
            "review_count": rng.randint(10, 5000)
        })
    return {"products": products, "total": len(products)}


@app.get("/items/{item_id}")
def get_item_detail(item_id: str, category: str = "electronics"):
    """Lấy thông tin chi tiết của 1 sản phẩm từ Qdrant payload."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(item_id)))
    results  = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        raise HTTPException(status_code=404, detail="Sản phẩm không tồn tại.")
    payload = results[0].payload
    
    import random
    iid = payload.get("item_id", item_id)
    rng = random.Random(iid)
    mock_title, mock_image = get_mock_title_and_image(category, iid, rng)
    
    original_title = payload.get("title", "")
    if original_title:
        context = original_title + " | " + payload.get("context", "")
    else:
        context = payload.get("context", "")

    return {
        "item_id": iid,
        "title":   mock_title,
        "context": context,
        "image":   mock_image,
        "rating":  round(rng.uniform(3.5, 5.0), 1),
        "review_count": rng.randint(10, 5000),
        "reviews": [],  # Có thể bổ sung từ Bronze Layer sau
    }
