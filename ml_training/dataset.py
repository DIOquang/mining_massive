import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class UserTowerDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # 1. Trích xuất đặc trưng User và điền 0 cho các giá trị rỗng
        self.features = df[[
            "total_reviews", 
            "avg_rating_given", 
            "stddev_rating_given"
        ]].fillna(0).astype(np.float32).values
        
        # 2. Lấy nhãn (Label: 1 hoặc 0)
        self.labels = df["label"].astype(np.float32).values
        
        # 3. Lấy trực tiếp cột embedding (Đã được gộp sẵn từ DuckDB)
        # Ép kiểu list/array thành numpy array cho nhanh
        self.item_embeddings = df["embedding"].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        user_feat = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        
        # Parse embedding trực tiếp từ row
        item_vector = torch.tensor(self.item_embeddings[idx], dtype=torch.float32)
        
        return user_feat, item_vector, label