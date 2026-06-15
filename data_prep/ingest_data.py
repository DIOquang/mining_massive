import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np
from google.cloud import firestore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
PROJECT_ID        = "project-528e2858-1a08-4d22-bcd"
GCS_EMBEDDINGS    = "gs://amazon-reviews-lakehouse-warehouse/warehouse/gold/item_embeddings"
GCS_ITEM_FEATURES = "gs://amazon-reviews-lakehouse-warehouse/warehouse/gold/item_features_v2"
GCS_REVIEWS       = "gs://amazon-reviews-lakehouse-warehouse/warehouse/silver"

QDRANT_HOST       = "34.41.86.165"
QDRANT_PORT       = 6333
COLLECTION_NAME   = "amazon_items"
VECTOR_DIM        = 384

SUBSET_SIZE       = 200_000   # số sản phẩm nạp vào Qdrant
FIRESTORE_REVIEWS = 5         # số bình luận lưu mỗi sản phẩm

# ─── KHỞI TẠO CLIENT ─────────────────────────────────────────────────────────
db     = firestore.Client(project=PROJECT_ID)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)


def create_qdrant_collection():
    """Tạo collection trong Qdrant nếu chưa tồn tại."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        log.info(f"Collection '{COLLECTION_NAME}' đã tồn tại, bỏ qua bước tạo.")
        return
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"✅ Đã tạo collection '{COLLECTION_NAME}'")


def load_embeddings_subset() -> dict:
    """Đọc ngẫu nhiên SUBSET_SIZE embeddings từ GCS Parquet."""
    log.info(f"⏳ Đang đọc embeddings từ GCS: {GCS_EMBEDDINGS}")
    dataset = ds.dataset(GCS_EMBEDDINGS, format="parquet")
    total   = dataset.count_rows()
    log.info(f"  Tổng số items trong GCS: {total:,}")

    # Lấy ngẫu nhiên SUBSET_SIZE chỉ số
    indices = sorted(random.sample(range(total), min(SUBSET_SIZE, total)))

    # Đọc từng batch, lọc theo chỉ số
    item_ids   = []
    embeddings = []
    seen = 0
    idx_set = set(indices)

    for batch in dataset.to_batches(columns=["item_id", "embedding"]):
        n = len(batch)
        for local_i in range(n):
            global_i = seen + local_i
            if global_i in idx_set:
                item_ids.append(batch.column("item_id")[local_i].as_py())
                embeddings.append(batch.column("embedding")[local_i].as_py())
        seen += n
        if len(item_ids) >= SUBSET_SIZE:
            break

    log.info(f"✅ Đã tải {len(item_ids):,} embeddings")
    return {"item_ids": item_ids, "embeddings": embeddings}


def load_item_metadata(item_ids: list) -> dict:
    """Đọc metadata (tên, danh mục) từ Gold Layer item_features."""
    log.info(f"⏳ Đang đọc metadata sản phẩm từ GCS: {GCS_ITEM_FEATURES}")
    dataset = ds.dataset(GCS_ITEM_FEATURES, format="parquet")
    item_set = set(item_ids)

    meta = {}
    for batch in dataset.to_batches(columns=["item_id", "item_text_context"]):
        for i in range(len(batch)):
            iid = batch.column("item_id")[i].as_py()
            if iid in item_set:
                meta[iid] = {
                    "item_text_context": batch.column("item_text_context")[i].as_py() or ""
                }
        if len(meta) >= len(item_ids):
            break

    log.info(f"✅ Đã đọc metadata cho {len(meta):,} sản phẩm")
    return meta


def ingest_to_qdrant(item_ids: list, embeddings: list, metadata: dict):
    """Nạp vectors + payload vào Qdrant theo batch."""
    log.info("⏳ Đang nạp vectors vào Qdrant...")
    BATCH = 1000
    total_uploaded = 0

    for start in range(0, len(item_ids), BATCH):
        batch_ids  = item_ids[start:start + BATCH]
        batch_embs = embeddings[start:start + BATCH]

        points = []
        for iid, emb in zip(batch_ids, batch_embs):
            meta = metadata.get(iid, {})
            # Lấy tên sản phẩm từ item_text_context (dòng đầu tiên)
            ctx   = meta.get("item_text_context", "")
            title = ctx.split("\n")[0][:200] if ctx else iid
            points.append(PointStruct(
                id      = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(iid))),
                vector  = emb,
                payload = {
                    "item_id": iid,
                    "title":   title,
                    "context": ctx[:500],
                }
            ))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total_uploaded += len(points)
        if total_uploaded % 10_000 == 0 or total_uploaded == len(item_ids):
            log.info(f"  Qdrant: {total_uploaded:,}/{len(item_ids):,} vectors đã nạp")

    log.info(f"✅ Hoàn tất nạp {total_uploaded:,} vectors vào Qdrant!")


def ingest_to_firestore(item_ids: list, metadata: dict):
    """Lưu metadata sản phẩm lên Firestore để Frontend hiển thị."""
    log.info("⏳ Đang đẩy metadata lên Firestore...")
    BATCH_SIZE = 500
    col = db.collection("items")
    total = 0

    for start in range(0, len(item_ids), BATCH_SIZE):
        batch_ref = db.batch()
        for iid in item_ids[start:start + BATCH_SIZE]:
            meta  = metadata.get(iid, {})
            ctx   = meta.get("item_text_context", "")
            title = ctx.split("\n")[0][:200] if ctx else str(iid)
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(iid)))
            batch_ref.set(col.document(doc_id), {
                "item_id":  iid,
                "title":    title,
                "context":  ctx[:1000],
            })
        batch_ref.commit()
        total += min(BATCH_SIZE, len(item_ids) - start)
        log.info(f"  Firestore: {total:,}/{len(item_ids):,} sản phẩm đã lưu")

    log.info(f"✅ Hoàn tất đẩy {total:,} sản phẩm lên Firestore!")


def main():
    log.info("═" * 60)
    log.info("🚀 BẮT ĐẦU QUÁ TRÌNH INGEST DỮ LIỆU")
    log.info("═" * 60)

    # Bước 1: Tạo collection Qdrant
    create_qdrant_collection()

    # Bước 2: Tải embeddings từ GCS
    data = load_embeddings_subset()
    item_ids   = data["item_ids"]
    embeddings = data["embeddings"]

    # Bước 3: Tải metadata
    metadata = load_item_metadata(item_ids)

    # Bước 4: Nạp vào Qdrant
    ingest_to_qdrant(item_ids, embeddings, metadata)

    # Bước 5: Đẩy lên Firestore
    ingest_to_firestore(item_ids, metadata)

    log.info("═" * 60)
    log.info("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH INGEST!")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
