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
    print("⏳ Bước 1: Đang tải Dữ liệu huấn luyện đã được gộp sẵn (Pre-joined)...")
    # Đọc trực tiếp từ thư mục gốc, bỏ qua file extract bị lỗi
    train_path = "/teamspace/studios/this_studio/local_data/ready_to_train"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu huấn luyện tại {train_path}")
        
    print("   ⚠️ Đang trích xuất tối đa 2 triệu dòng để tránh tràn bộ nhớ (OOM)...")
    # Sử dụng dataset và head() để đọc một phần dữ liệu, tránh tràn RAM nếu file parquet quá khổng lồ
    dataset = ds.dataset(train_path, format="parquet")
    df_train = dataset.head(2_000_000).to_pandas()
    print(f"   ✅ Đã nạp {len(df_train)} dòng dữ liệu tương tác (đã kèm sẵn embedding).")
    
    return df_train

def main():
    # Ép kiểu float32 matmul về mức 'medium' để tận dụng tối đa kiến trúc phần cứng GPU L4
    torch.set_float32_matmul_precision('medium')

    # 1. Tải dữ liệu siêu nhẹ (đã gộp sẵn)
    df_train = load_local_data_optimized()
    
    # 2. Đóng gói dữ liệu
    train_dataset = UserTowerDataset(df_train)
    
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
        max_steps=5000,       # DỪNG SỚM sau 5.000 bước (Khoảng vài tiếng tùy GPU) để tiết kiệm Credit
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