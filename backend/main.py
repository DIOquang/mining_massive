from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import redis
import json
import hashlib
import logging
from typing import List
from qdrant_client import QdrantClient
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel

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
vertexai.init(project=PROJECT_ID, location=REGION)
llm    = GenerativeModel("gemini-1.5-pro")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
db     = firestore.Client(project=PROJECT_ID)

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


def rerank_with_gemini(session_context: str, candidates: list) -> List[dict]:
    """Dùng Vertex AI Gemini để Rerank Top-10 và sinh giải thích."""
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

Hãy chọn ra đúng {TOP_K_FINAL} sản phẩm phù hợp nhất và trả lời theo định dạng JSON sau (chỉ JSON, không giải thích thêm):
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

    response = llm.generate_content(prompt)
    text     = response.text.strip()

    # Làm sạch JSON (đôi khi Gemini thêm ```json ... ```)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    data = json.loads(text)
    return data.get("recommendations", [])


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

    # 6. Rerank bằng Gemini Vertex AI
    reranked = rerank_with_gemini(session_context, candidates)

    # 7. Ghép điểm score từ Qdrant
    score_map = {c.payload["item_id"]: c.score for c in candidates}
    result = []
    for r in reranked:
        result.append(RecommendItem(
            item_id     = r["item_id"],
            title       = r["title"],
            score       = score_map.get(r["item_id"], 0.0),
            explanation = r["explanation"],
        ))

    # 8. Lưu cache (TTL 1 giờ)
    if cache:
        cache.setex(cache_key, 3600, json.dumps([r.dict() for r in result]))

    return RecommendResponse(
        recommendations=result,
        session_size=len(req.clicked_item_ids),
        from_cache=False,
    )


@app.get("/items/{item_id}")
def get_item_detail(item_id: str):
    """Lấy thông tin chi tiết + reviews của 1 sản phẩm từ Firestore."""
    import uuid
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item_id))
    doc    = db.collection("items").document(doc_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Sản phẩm không tồn tại.")
    return doc.to_dict()
