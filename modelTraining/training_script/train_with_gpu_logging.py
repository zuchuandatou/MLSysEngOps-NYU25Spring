import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import GPUtil
import threading
import time

class RatingDataset(Dataset):
    def __init__(self, npz_dir):
        self.user_list, self.item_list = [], []
        for npz_path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
            data = np.load(npz_path)
            arr = data["arr_0"]
            self.user_list.append(torch.LongTensor(arr[:, 0]))
            self.item_list.append(torch.LongTensor(arr[:, 1]))
        self.users = torch.cat(self.user_list)
        self.items = torch.cat(self.item_list)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, users, items):
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores

def log_gpu_usage(interval=10):
    while True:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            mlflow.log_metric(f"gpu_{i}_load", gpu.load)
            mlflow.log_metric(f"gpu_{i}_mem_used_MB", gpu.memoryUsed)
        time.sleep(interval)

def train(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users, items = users.to(device), items.to(device)
            labels = torch.ones_like(users, dtype=torch.float32).to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                labels = torch.ones_like(users, dtype=torch.float32).to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(labels)

        avg_val_loss = val_loss / len(val_loader.dataset)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

    return model

if __name__ == "__main__":
    MOVIELENS20M_DATA_DIR = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_dir = os.path.join(MOVIELENS20M_DATA_DIR, "training")
    val_dir = os.path.join(MOVIELENS20M_DATA_DIR, "validation")

    train_dataset = RatingDataset(train_dir)
    val_dataset = RatingDataset(val_dir)

    num_users = max(train_dataset.users.max().item(), val_dataset.users.max().item()) + 1
    num_items = max(train_dataset.items.max().item(), val_dataset.items.max().item()) + 1

    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8192, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run():
        mlflow.log_param("embedding_size", 64)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("batch_size", 8192)
        mlflow.log_param("epochs", 5)

        # 启动 GPU 日志线程
        gpu_logger = threading.Thread(target=log_gpu_usage, args=(10,), daemon=True)
        gpu_logger.start()

        model = RecommenderNet(num_users, num_items, embedding_size=64)
        model = train(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("✅ Training complete and model logged.")
