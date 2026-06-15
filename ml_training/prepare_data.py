import duckdb
import os
import shutil

def merge_data():
    output_dir = "local_data/ready_to_train"
    
    # --- CẬP NHẬT TỰ ĐỘNG DỌN DẸP ---
    if os.path.exists(output_dir):
        print(f"🧹 Phát hiện dữ liệu cũ trong '{output_dir}', đang dọn dẹp làm sạch...")
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ Đang tiến hành ghép dữ liệu bằng DuckDB...")
    print("Thao tác này chạy trên ổ cứng, có thể mất vài phút nhưng chỉ cần làm 1 lần duy nhất.")

    # Kích hoạt thanh tiến độ
    duckdb.sql("SET enable_progress_bar=true;")
    duckdb.sql("SET progress_bar_time=100;") 
    
    # Chạy truy vấn gộp
    query = f"""
    COPY (
        SELECT 
            t.total_reviews, 
            t.avg_rating_given, 
            t.stddev_rating_given, 
            t.label, 
            e.embedding
        FROM read_parquet('local_data/training_data/*.parquet') AS t
        INNER JOIN read_parquet('local_data/item_embeddings/*.parquet') AS e
        ON t.item_id = e.item_id
    ) TO '{output_dir}' (FORMAT PARQUET, FILE_SIZE_BYTES 100000000);
    """
    
    duckdb.sql(query)
    print(f"\n✅ Đã chuẩn bị xong! Dữ liệu huấn luyện hoàn chỉnh nằm tại: {output_dir}/")

if __name__ == "__main__":
    merge_data()