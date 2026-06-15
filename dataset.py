import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class UserTowerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, item_embeddings_dict: dict):
        # 1. Trích xuất đặc trưng User và điền 0 cho các giá trị rỗng
        self.features = df[[
            "total_reviews", 
            "avg_rating_given", 
            "stddev_rating_given"
        ]].fillna(0).astype(np.float32).values
        
        # 2. Lấy nhãn (Label: 1 hoặc 0)
        self.labels = df["label"].astype(np.float32).values
        
        # 3. Lấy danh sách Item ID tương ứng để tra cứu vector
        self.item_ids = df["item_id"].values
        self.item_embeddings = item_embeddings_dict
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        user_feat = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        
        # Lấy vector item (384 chiều), nếu không tìm thấy sẽ trả về vector toàn số 0
        item_id = self.item_ids[idx]
        item_vector = torch.tensor(
            self.item_embeddings.get(item_id, np.zeros(384, dtype=np.float32))
        )
        
        return user_feat, item_vector, label