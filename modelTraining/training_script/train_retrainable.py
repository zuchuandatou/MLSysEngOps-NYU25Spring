import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import pynvml
from glob import glob

class RatingDataset(Dataset):
    def __init__(self, npz_files):
        user_list = []
        item_list = []
        for npz_path in npz_files:
            data = np.load(npz_path)["arr_0"]
            user_list.append(torch.LongTensor(data[:, 0]))
            item_list.append(torch.LongTensor(data[:, 1]))
        self.users = torch.cat(user_list)
        self.items = torch.cat(item_list)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return scores

def get_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024**2  # MB

def train(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users, items = users.to(device), items.to(device)
            labels = torch.ones_like(users, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(users, items)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(users)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("gpu_mem_MB", get_gpu_usage(), step=epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                labels = torch.ones_like(users, dtype=torch.float32).to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(users, items)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * len(users)

        avg_val_loss = val_loss / len(val_loader.dataset)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    return model

if __name__ == "__main__":
    mlflow.set_experiment("MovieLens20M-Retrain")

    data_root = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_dir = os.path.join(data_root, "training")
    val_dir = os.path.join(data_root, "validation")

    train_files = sorted(glob(os.path.join(train_dir, "*.npz")))[:4]
    val_files = sorted(glob(os.path.join(val_dir, "*.npz")))[:2]

    train_data = RatingDataset(train_files)
    val_data = RatingDataset(val_files)

    num_users = max(train_data.users.max(), val_data.users.max()).item() + 1
    num_items = max(train_data.items.max(), val_data.items.max()).item() + 1

    train_loader = DataLoader(train_data, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8192)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retrain = os.getenv("RETRAIN", "false").lower() == "true"
    model_path = os.getenv("RETRAIN_MODEL_PATH", "SSE_PT_tf.pth")

    model = RecommenderNet(num_users, num_items, embedding_size=64)
    if retrain and os.path.exists(model_path):
        print(f"🔁 Loading model from {model_path} for retraining...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("🆕 Training model from scratch.")

    with mlflow.start_run(run_name="MovieLens20M-Retrain"):
        mlflow.log_param("embedding_size", 64)
        mlflow.log_param("batch_size", 8192)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("retrain", retrain)
        mlflow.log_param("model_path", model_path)

        model = train(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact("model.pt")
