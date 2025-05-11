import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class ImplicitDataset(Dataset):
    def __init__(self, npz_dir):
        self.data = []
        for path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
            arr = np.load(path)['arr_0']
            self.data.append(torch.LongTensor(arr))
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]  # user, item


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return torch.sigmoid(scores)  # output: probability of interaction


def train(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users, items = users.to(device), items.to(device)
            labels = torch.ones_like(users, dtype=torch.float32, device=device)  # implicit: all ones
            preds = model(users, items)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(users)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items in val_loader:
                    users, items = users.to(device), items.to(device)
                    labels = torch.ones_like(users, dtype=torch.float32, device=device)
                    preds = model(users, items)
                    val_loss += loss_fn(preds, labels).item() * len(users)

            avg_val_loss = val_loss / len(val_loader.dataset)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model


if __name__ == "__main__":
    MOVIELENS20M_DATA_DIR = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_dir = os.path.join(MOVIELENS20M_DATA_DIR, "training")
    val_dir = os.path.join(MOVIELENS20M_DATA_DIR, "validation")

    train_data = ImplicitDataset(train_dir)
    val_data = ImplicitDataset(val_dir)

    num_users = max(torch.max(train_data[:][0]), torch.max(val_data[:][0])).item() + 1
    num_items = max(torch.max(train_data[:][1]), torch.max(val_data[:][1])).item() + 1

    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2048)

    with mlflow.start_run():
        mlflow.log_param("embedding_size", 64)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("batch_size", 2048)
        mlflow.log_param("epochs", 5)

        model = RecommenderNet(num_users, num_items, embedding_size=64)
        model = train(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("✅ Training complete and logged to MLflow")