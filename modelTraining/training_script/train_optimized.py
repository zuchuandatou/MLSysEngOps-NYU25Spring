import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch


class ImplicitDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
            arr = np.load(path)['arr_0']
            self.data.append(torch.LongTensor(arr))
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1]


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):  # 更小 embedding 加快训练
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, users, items):
        users = users.clamp(max=self.user_embedding.num_embeddings - 1)
        items = items.clamp(max=self.item_embedding.num_embeddings - 1)
        user_vecs = self.user_embedding(users)
        item_vecs = self.item_embedding(items)
        scores = (user_vecs * item_vecs).sum(dim=1)
        return torch.sigmoid(scores)


def train(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users, items = users.to(device), items.to(device)
            preds = model(users, items)
            labels = torch.ones_like(preds)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(users)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        val_loss = 0
        hit_count = 0
        total_count = 0
        with torch.no_grad():
            for users, items in val_loader:
                users, items = users.to(device), items.to(device)
                preds = model(users, items)
                labels = torch.ones_like(preds)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(users)

                # binary accuracy (>0.5 is considered a hit)
                hit_count += (preds > 0.5).sum().item()
                total_count += len(preds)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = hit_count / total_count
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    return model


if __name__ == "__main__":
    data_dir = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")

    # Faster training: sample subset
    full_train_data = ImplicitDataset(train_dir)
    subset_ratio = 0.3
    train_size = int(len(full_train_data) * subset_ratio)
    train_subset, _ = random_split(full_train_data, [train_size, len(full_train_data) - train_size])

    val_data = ImplicitDataset(val_dir)

    user_count = int(max(train_subset[:][0].max(), val_data[:][0].max()).item()) + 1
    item_count = int(max(train_subset[:][1].max(), val_data[:][1].max()).item()) + 1

    BATCH_SIZE = 1024  # 更小 batch 避免 OOM
    EPOCHS = 5
    LR = 5e-4
    EMBED_DIM = 32

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run():
        mlflow.set_tag("model", "MatrixFactorization")
        mlflow.log_param("embedding_dim", EMBED_DIM)
        mlflow.log_param("lr", LR)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("train_subset_ratio", subset_ratio)

        model = MF(user_count, item_count, embedding_dim=EMBED_DIM)
        model = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("✅ Training complete and logged to MLflow.")