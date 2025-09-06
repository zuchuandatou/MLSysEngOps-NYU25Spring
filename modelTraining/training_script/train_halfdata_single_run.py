import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import GPUtil
import threading
import time

class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, npz_dir):
        self.data = []
        for npz_file in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
            arr = np.load(npz_file)["arr_0"]
            self.data.append(torch.tensor(arr[:, :2], dtype=torch.long))
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1]

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, users, items):
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        return (user_vecs * item_vecs).sum(dim=1)

def log_gpu(interval=10):
    while True:
        gpus = GPUtil.getGPUs()
        for i, g in enumerate(gpus):
            mlflow.log_metric(f"gpu_{i}_load", g.load)
            mlflow.log_metric(f"gpu_{i}_mem", g.memoryUsed)
        time.sleep(interval)

def train(model, train_loader, val_loader, epochs, lr, device):
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
            total_loss += loss.item() * len(users)
        mlflow.log_metric("train_loss", total_loss / len(train_loader.dataset), step=epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                labels = torch.ones_like(users, dtype=torch.float32).to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(users)
        mlflow.log_metric("val_loss", val_loss / len(val_loader.dataset), step=epoch)

    return model

if __name__ == "__main__":
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    mlflow.autolog(disable=True)

    data_root = os.getenv("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_dir = os.path.join(data_root, "training")
    val_dir = os.path.join(data_root, "validation")

    full_train = RatingDataset(train_dir)
    subset_len = len(full_train) // 2
    train_data, _ = random_split(full_train, [subset_len, len(full_train) - subset_len])
    val_data = RatingDataset(val_dir)

    num_users = max(full_train.data[:, 0].max(), val_data.data[:, 0].max()).item() + 1
    num_items = max(full_train.data[:, 1].max(), val_data.data[:, 1].max()).item() + 1

    train_loader = DataLoader(train_data, batch_size=8192, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=8192, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(nested=False):
        threading.Thread(target=log_gpu, daemon=True).start()
        mlflow.log_params({
            "embedding_dim": 32,
            "batch_size": 8192,
            "epochs": 3,
            "lr": 1e-3,
            "train_split": 0.5
        })

        model = RecommenderNet(num_users, num_items, embedding_dim=32)
        model = train(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)
        mlflow.pytorch.log_model(model, "model")
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "movielens_implicit")
        print("✅ Training complete with half data and single MLflow run.")