import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import argparse

# GPU 监控（可选）
try:
    import pynvml
    pynvml.nvmlInit()
    def get_gpu_memory():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
except ImportError:
    def get_gpu_memory():
        return 0

# 数据集加载
class RatingDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)["arr_0"]
        self.users = torch.LongTensor(data[:, 0])
        self.items = torch.LongTensor(data[:, 1])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

# 推荐模型定义
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

    def forward(self, users, items=None):
        user_vecs = self.user_emb(users)
        if items is not None:
            item_vecs = self.item_emb(items)
            return (user_vecs * item_vecs).sum(dim=1)
        else:
            item_vecs = self.item_emb.weight
            return torch.matmul(user_vecs, item_vecs.T)

def train(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for users, items in pbar:
            users, items = users.to(device), items.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(users)
                loss = loss_fn(logits, items)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), gpu_mem=f"{get_gpu_memory():.0f}MB")

        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)

        # 验证
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                logits = model(users)
                loss = loss_fn(logits, items)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == items).sum().item()
                total += len(items)
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
        mlflow.log_metric("val_accuracy", val_correct / total, step=epoch)

    return model

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000"))
    mlflow.set_experiment("MovieLens20M-Retrain")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = "/mnt/MovieLens-20M/organized/training"
    val_dir = "/mnt/MovieLens-20M/organized/validation"
    eval_dir = "/mnt/MovieLens-20M/organized/evaluation"

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npz")]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".npz")]
    eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".npz")]

    full_train = torch.utils.data.ConcatDataset([RatingDataset(f) for f in train_files])
    full_val = torch.utils.data.ConcatDataset([RatingDataset(f) for f in val_files])
    full_eval = torch.utils.data.ConcatDataset([RatingDataset(f) for f in eval_files])

    # 仅使用 1/10 数据
    tenth_len_train = len(full_train) // 10
    tenth_len_val = len(full_val) // 10
    tenth_len_eval = len(full_eval) // 10
    train_data, _ = random_split(full_train, [tenth_len_train, len(full_train) - tenth_len_train])
    val_data, _ = random_split(full_val, [tenth_len_val, len(full_val) - tenth_len_val])
    eval_data, _ = random_split(full_eval, [tenth_len_eval, len(full_eval) - tenth_len_eval])

    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=2048)
    eval_loader = DataLoader(eval_data, batch_size=2048)

    num_users = max(train_data.dataset.datasets[0].users.max().item(), val_data.dataset.datasets[0].users.max().item()) + 1
    num_items = max(train_data.dataset.datasets[0].items.max().item(), val_data.dataset.datasets[0].items.max().item()) + 1

    with mlflow.start_run():
        mlflow.log_params({"emb_size": 64, "lr": 1e-3, "batch_size": 2048, "epochs": 5})
        model = MatrixFactorization(num_users, num_items)
        if args.retrain and os.path.exists("SSE_PT_tf.pth"):
            model.load_state_dict(torch.load("SSE_PT_tf.pth"))
            print("Loaded pretrained weights.")
        else:
            print("Training model from scratch.")
        model = train(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)
        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), "model_final.pth")
        print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Use pretrained model")
    args = parser.parse_args()
    main(args)
