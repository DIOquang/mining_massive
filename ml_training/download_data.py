import gcsfs
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_folder_with_progress(fs, gcs_folder, local_folder, max_workers=8):
    print(f"🔍 Đang quét danh sách file trong: {gcs_folder}...")
    
    # Lấy danh sách toàn bộ các file bên trong thư mục GCS
    all_files = fs.find(gcs_folder)
    print(f"📋 Tìm thấy tổng cộng {len(all_files)} files. Kích hoạt {max_workers} luồng tải song song...\n")

    def download_single_file(gcs_file_path):
        # Tính toán đường dẫn lưu file trên ổ cứng máy ảo
        relative_path = os.path.relpath(gcs_file_path, gcs_folder)
        local_file_path = os.path.join(local_folder, relative_path)
        
        # Tạo thư mục con nếu có
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Tiến hành kéo file
        fs.get(gcs_file_path, local_file_path)
        return gcs_file_path

    # Hiển thị thanh Progress Bar và tải đa luồng
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_file, f) for f in all_files]
        
        # Dùng tqdm để bọc danh sách tác vụ đang chạy
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"📥 Đang tải {os.path.basename(gcs_folder)}", unit="file", colour="green"):
            pass

def main():
    print("⏳ Đang kết nối tới Google Cloud Storage bằng quyền của Python...")
    fs = gcsfs.GCSFileSystem()

    # 1. Tải Item Embeddings (14GB)
    download_folder_with_progress(
        fs=fs,
        gcs_folder="amazon-reviews-lakehouse-warehouse/warehouse/gold/item_embeddings",
        local_folder="local_data/item_embeddings",
        max_workers=8 # Tốc độ xé gió với 8 luồng song song
    )
    print("✅ Đã kéo xong toàn bộ Item Embeddings!\n")

    # 2. Tải Training Data
    download_folder_with_progress(
        fs=fs,
        gcs_folder="amazon-reviews-lakehouse-warehouse/warehouse/gold/training_data",
        local_folder="local_data/training_data",
        max_workers=4
    )
    print("✅ Đã kéo xong toàn bộ Dữ liệu huấn luyện!")

if __name__ == "__main__":
    main()