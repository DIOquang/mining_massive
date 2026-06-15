import pytorch_lightning as pl
import torch
import torch.nn as nn

class UserTowerLightning(pl.LightningModule):
    def __init__(self, input_dim=3, output_dim=384, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Cấu trúc mạng MLP (Đã được nâng cấp để vắt kiệt sức mạnh GPU và học sâu hơn)
        # Tăng số lượng tham số từ 50K lên khoảng 4.7 Triệu tham số.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, output_dim)
        )
        
        # Hàm loss đánh giá phân loại nhị phân
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, user_features):
        return self.mlp(user_features)

    def training_step(self, batch, batch_idx):
        user_feat, item_vector, labels = batch
        
        # Đi qua mạng để lấy vector của User
        user_vector = self(user_feat)
        
        # Tương tác Dot Product giữa User và Item
        logits = (user_vector * item_vector).sum(dim=1)
        
        # Tính Loss
        loss = self.loss_fn(logits, labels)
        
        # Ghi log
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return optimizer