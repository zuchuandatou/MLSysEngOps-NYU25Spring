
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from torch.cuda.amp import autocast, GradScaler

# ===================== Dataset =====================
class NpzRatingDataset(Dataset):
    def __init__(self, npz_files):
        self.data = []
        for f in npz_files:
            arr = np.load(f)["arr_0"]
            self.data.append(torch.tensor(arr, dtype=torch.long))
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item = self.data[idx]
        return user, item

# ===================== Model =====================
class SSEPT(nn.Module):
    def __init__(self, num_users, num_items, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.user_embedding_layer = nn.Embedding(num_users, d_model)
        self.item_embedding_layer = nn.Embedding(num_items, d_model)
        self.positional_embedding_layer = nn.Parameter(torch.randn(1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, num_items)

    def forward(self, user_ids):
        user_emb = self.user_embedding_layer(user_ids)
        x = user_emb + self.positional_embedding_layer
        x = self.encoder(x)
        return self.fc_out(x)

# ===================== Training =====================
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0
    for user_ids, item_ids in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(user_ids)
            loss = loss_fn(logits, item_ids)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_ids, item_ids in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            logits = model(user_ids)
            loss = loss_fn(logits, item_ids)
            total_loss += loss.item()
    return total_loss / len(loader)

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_files = [os.path.join(data_root, "training", f) for f in sorted(os.listdir(os.path.join(data_root, "training"))) if f.endswith(".npz")]
    val_files = [os.path.join(data_root, "validation", f) for f in sorted(os.listdir(os.path.join(data_root, "validation"))) if f.endswith(".npz")]

    train_dataset = NpzRatingDataset(train_files[:max(1, len(train_files)//10)])
    val_dataset = NpzRatingDataset(val_files[:max(1, len(val_files)//10)])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    num_users = train_dataset.data[:, 0].max().item() + 1
    num_items = train_dataset.data[:, 1].max().item() + 1

    model = SSEPT(num_users, num_items).to(device)
    if args.retrain:
        state_dict = torch.load("SSE_PT_tf.pth")
        for key in list(state_dict.keys()):
            if "user_embedding_layer" in key or "item_embedding_layer" in key:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    mlflow.set_experiment("MovieLens20M-Retrain")
    with mlflow.start_run():
        mlflow.log_params({"lr": 1e-3, "batch_size": 1024, "d_model": 64})

        for epoch in range(1, 6):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
            val_loss = evaluate(model, val_loader, loss_fn, device)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        torch.save(model.state_dict(), "SSE_PT_tf_retrained.pth")
        mlflow.log_artifact("SSE_PT_tf_retrained.pth")

if __name__ == "__main__":
    main()
