import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

# Đảm bảo các file dataset.py và model.py nằm cùng thư mục này
from dataset import UserTowerDataset
from model import UserTowerLightning

def load_local_data_optimized():
    print("⏳ Bước 1: Đang tải Dữ liệu huấn luyện từ Ổ CỨNG LOCAL...")
    train_path = "local_data/training_data/part-00000.parquet"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu huấn luyện tại {train_path}")
        
    df_train = pq.read_table(train_path).to_pandas()
    print(f"   ✅ Đã nạp {len(df_train)} dòng dữ liệu tương tác.")
    
    needed_item_ids = set(df_train["item_id"].unique())
    print(f"   📋 Tìm thấy {len(needed_item_ids)} sản phẩm độc nhất cần lấy Vector.")

    print("\n⏳ Bước 2: Quét tập 14GB Embeddings cục bộ (Lazy-evaluation)...")
    emb_path = "local_data/item_embeddings"
    
    dataset = ds.dataset(emb_path, format="parquet")
    item_embeddings_dict = {}
    
    # Đọc tối ưu: Giữ nguyên định dạng C++ Array của PyArrow trong bộ nhớ
    for batch in dataset.to_batches(columns=["item_id", "embedding"]):
        ids = batch.column("item_id").to_pylist()
        # TUYỆT ĐỐI KHÔNG dùng .to_pylist() trên cột embedding ở đây nữa
        emb_column = batch.column("embedding")
        
        for idx, item_id in enumerate(ids):
            if item_id in needed_item_ids:
                # SỬA LỖI: Chỉ ép kiểu sang Python Object duy nhất cho phần tử trúng tuyển
                item_embeddings_dict[item_id] = np.array(emb_column[idx].as_py(), dtype=np.float32)
                
    print(f"   ✅ Đã nạp thành công {len(item_embeddings_dict)} vectors phù hợp vào RAM. Bộ nhớ an toàn!")
    return df_train, item_embeddings_dict

def main():
    # Ép kiểu float32 matmul về mức 'medium' để tận dụng tối đa kiến trúc phần cứng GPU L4
    torch.set_float32_matmul_precision('medium')

    # 1. Tải dữ liệu siêu nhẹ
    df_train, item_embeddings_dict = load_local_data_optimized()
    
    # 2. Đóng gói dữ liệu
    train_dataset = UserTowerDataset(df_train, item_embeddings_dict)
    
    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4096,      
        shuffle=True, 
        num_workers=4,        # Giảm xuống 4 workers để giảm tải bộ nhớ cấp phát RAM của CPU
        pin_memory=True,      
        drop_last=True        
    )

    # 4. Khởi tạo Mô hình User Tower
    print("\n⚙️ Đang khởi tạo kiến trúc Tháp Người Dùng...")
    model = UserTowerLightning(input_dim=3, output_dim=384, lr=0.001)

    # 5. Cấu hình Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints/",
        filename="user_tower-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,         
        monitor="train_loss",
        mode="min"
    )

    # 6. Khởi tạo Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=10,        
        accelerator="gpu",
        devices="auto",
        precision="16-mixed", 
        log_every_n_steps=10, 
        callbacks=[checkpoint_callback]
    )

    # 7. Phóng!
    print("\n🚀 BẮT ĐẦU PHÓNG MÔ HÌNH LÊN HỆ THỐNG GPU L4...")
    trainer.fit(model, train_dataloaders=train_loader)
    
    print(f"\n🎉 HOÀN TẤT! Mô hình xuất sắc nhất đã được lưu an toàn tại:\n{checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()