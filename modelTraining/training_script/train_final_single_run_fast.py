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
        self.samples = []
        for npz_path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
            arr = np.load(npz_path)["arr_0"]
            self.samples.append(torch.tensor(arr[:, :2], dtype=torch.long))
        self.data = torch.cat(self.samples, dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1]

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, users, items):
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        return (user_vecs * item_vecs).sum(dim=1)

def log_gpu_usage(interval=10):
    while True:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            mlflow.log_metric(f"gpu_{i}_load", gpu.load)
            mlflow.log_metric(f"gpu_{i}_mem_MB", gpu.memoryUsed)
        time.sleep(interval)

def train(model, train_loader, val_loader, epochs=3, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            users, items = users.to(device), items.to(device)
            labels = torch.ones_like(users, dtype=torch.float32).to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(users)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                labels = torch.ones_like(users, dtype=torch.float32).to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                total_val_loss += loss.item() * len(users)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model

if __name__ == "__main__":
    data_root = os.getenv("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_path = os.path.join(data_root, "training")
    val_path = os.path.join(data_root, "validation")

    train_dataset = RatingDataset(train_path)
    val_dataset = RatingDataset(val_path)

    num_users = max(train_dataset.data[:, 0].max(), val_dataset.data[:, 0].max()).item() + 1
    num_items = max(train_dataset.data[:, 1].max(), val_dataset.data[:, 1].max()).item() + 1

    batch_size = 8192
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(nested=False):
        threading.Thread(target=log_gpu_usage, args=(10,), daemon=True).start()

        mlflow.set_tag("model", "RecommenderNet")
        mlflow.log_params({
            "embedding_size": 32,
            "batch_size": batch_size,
            "epochs": 3,
            "lr": 1e-3
        })

        model = RecommenderNet(num_users, num_items, embedding_size=32)
        model = train(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)

        torch.save(model.state_dict(), "recommender.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "movielens_implicit")
        print("✅ Training complete, model saved and registered.")