import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch


class RatingDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.users = torch.LongTensor(data['user'])
        self.items = torch.LongTensor(data['item'])
        self.ratings = torch.FloatTensor(data['rating'])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        return (user_vecs * item_vecs).sum(dim=1)

    def recommend(self, user_id, top_k=10):
        user_tensor = torch.LongTensor([user_id])
        all_items = torch.arange(self.item_embedding.num_embeddings)
        scores = self.forward(user_tensor.expand_as(all_items), all_items)
        top_items = torch.topk(scores, k=top_k).indices
        return top_items.tolist()


def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device="cpu"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            preds = model(users, items)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)

        avg_train_loss = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                preds = model(users, items)
                loss = criterion(preds, ratings)
                total_val_loss += loss.item() * len(ratings)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model


if __name__ == "__main__":
    data_root = os.environ.get("MOVIELENS20M_DATA_DIR", "/mnt/MovieLens-20M/organized")
    train_path = os.path.join(data_root, "training", "train_0.npz")
    val_path = os.path.join(data_root, "validation", "test_0.npz")

    train_data = RatingDataset(train_path)
    val_data = RatingDataset(val_path)

    user_count = int(max(train_data.users.max(), val_data.users.max()).item()) + 1
    item_count = int(max(train_data.items.max(), val_data.items.max()).item()) + 1

    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run():
        mlflow.log_param("embedding_size", 64)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("batch_size", 1024)
        mlflow.log_param("epochs", 5)

        model = RecommenderNet(user_count, item_count, embedding_size=64)
        model = train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("✅ Training and logging complete.")
