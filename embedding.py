import os
import sys
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

# --- CẤU HÌNH ---
GCP_PROJECT = "project-528e2858-1a08-4d22-bcd" # Nhớ kiểm tra lại Project ID
GCS_INPUT_DIR = "gs://amazon-reviews-lakehouse-warehouse/warehouse/gold/item_features_v2"
GCS_OUTPUT_DIR = "gs://amazon-reviews-lakehouse-warehouse/warehouse/gold/item_embeddings"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 512
CHUNK_SIZE = 100000 

# Khởi tạo Client và Bể luồng (Thread Pool) cho việc upload
storage_client = storage.Client(project=GCP_PROJECT)
upload_executor = ThreadPoolExecutor(max_workers=4)

def upload_worker_task(df_result, output_filepath):
    """Hàm chạy ngầm: Chịu trách nhiệm mang file đi upload để giải phóng GPU."""
    print(f"☁️ [Mạng] Đang upload ngầm lên: {output_filepath}")
    try:
        # Sử dụng zstd hoặc gzip nếu mạng chậm, snappy nếu mạng máy ảo cực khỏe
        df_result.to_parquet(
            output_filepath, 
            engine="pyarrow", 
            compression="snappy",
            index=False
        )
        print(f"✅ [Mạng] Upload thành công: {output_filepath}")
    except Exception as e:
        print(f"❌ [LỖI MẠNG] Upload thất bại file {output_filepath}: {e}")

def get_already_processed_ids():
    """Đọc tối ưu để lấy danh sách item_id đã embed mà không gây tràn RAM."""
    print("⏳ Kiểm tra các item đã được embed trước đó...")
    try:
        dataset = ds.dataset(GCS_OUTPUT_DIR, format="parquet")
        processed_set = set()
        
        # Đọc theo batch chỉ lấy cột item_id để tiết kiệm bộ nhớ
        for batch in dataset.to_batches(columns=["item_id"]):
            processed_set.update(batch.column("item_id").to_pylist())
            
        print(f"✅ Đã tìm thấy {len(processed_set)} items đã hoàn thành.")
        return processed_set
    except Exception as e:
        print("-> Thư mục Output trống hoặc chưa tồn tại. Bắt đầu từ con số 0.")
        return set()

def embed_and_upload_chunk(model, chunk_data, chunk_index):
    """Thực hiện nhúng vector bằng GPU và giao việc upload cho luồng ngầm."""
    print(f"\n⚡ [GPU] Đang chạy Embedding cho Chunk {chunk_index} ({len(chunk_data['item_id'])} items)...")
    
    embeddings = model.encode(
        chunk_data["text_to_embed"],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    df_result = pd.DataFrame({
        "item_id": chunk_data["item_id"],
        "embedding": embeddings.tolist()
    })
    
    output_filepath = f"{GCS_OUTPUT_DIR}/part-{chunk_index:05d}.parquet"
    
    # Fire and Forget: Đẩy data cho luồng ngầm xử lý, GPU ngay lập tức quay lại làm chunk mới
    upload_executor.submit(upload_worker_task, df_result, output_filepath)

def process_and_embed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"torch_dtype": torch.float16})
    processed_ids = list(get_already_processed_ids())
    
    print(f"⏳ Đang quét metadata từ: {GCS_INPUT_DIR}")
    dataset = ds.dataset(GCS_INPUT_DIR, format="parquet")
    
    # Lọc bằng C++ PyArrow từ lúc đọc dữ liệu để tiết kiệm RAM
    scan_filter = None
    if processed_ids:
        scan_filter = ~pc.field("item_id").isin(processed_ids)
    
    chunk_data = {"item_id": [], "text_to_embed": []}
    chunk_index = len(processed_ids) // CHUNK_SIZE
    
    try:
        for batch in dataset.to_batches(columns=["item_id", "item_text_context"], filter=scan_filter):
            df_batch = batch.to_pandas()
            if df_batch.empty:
                continue
                
            chunk_data["item_id"].extend(df_batch["item_id"].tolist())
            chunk_data["text_to_embed"].extend(df_batch["item_text_context"].fillna("No review available").tolist())
                
            while len(chunk_data["item_id"]) >= CHUNK_SIZE:
                current_chunk = {
                    "item_id": chunk_data["item_id"][:CHUNK_SIZE],
                    "text_to_embed": chunk_data["text_to_embed"][:CHUNK_SIZE]
                }
                
                embed_and_upload_chunk(model, current_chunk, chunk_index)
                chunk_index += 1
                
                chunk_data["item_id"] = chunk_data["item_id"][CHUNK_SIZE:]
                chunk_data["text_to_embed"] = chunk_data["text_to_embed"][CHUNK_SIZE:]
                    
        if len(chunk_data["item_id"]) > 0:
            embed_and_upload_chunk(model, chunk_data, chunk_index)

    except KeyboardInterrupt:
        print("\n\n⚠️ BẠN VỪA BẤM CTRL+C! KÍCH HOẠT QUY TRÌNH DỪNG AN TOÀN...")
        print("⏳ Đang chờ các file dở dang upload xong lên GCS. Vui lòng KHÔNG bấm Ctrl+C lần nữa!")
        upload_executor.shutdown(wait=True) 
        print("✅ Toàn bộ luồng mạng đã đóng an toàn. Hệ thống đã tắt!")
        sys.exit(0)
        
    # Đảm bảo upload xong những chunk cuối cùng nếu chương trình kết thúc tự nhiên
    print("\n⏳ Đang đợi các luồng upload cuối cùng hoàn tất...")
    upload_executor.shutdown(wait=True)

if __name__ == "__main__":
    process_and_embed()
    print("🎉 HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH EMBEDDING!")