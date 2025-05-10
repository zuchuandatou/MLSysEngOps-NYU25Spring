import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class CSVDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users   = torch.LongTensor(users)
        self.items   = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

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
        u = torch.LongTensor([user_id])
        all_items = torch.arange(self.item_embedding.num_embeddings)
        scores = self.forward(u.expand_as(all_items), all_items)
        return torch.topk(scores, k=top_k).indices.tolist()

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # —— 训练 —— #
        model.train()
        total_loss = 0.0
        for users, items, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            preds = model(users, items)
            loss = criterion(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ratings.size(0)

        avg_train = total_loss / len(train_loader.dataset)
        mlflow.log_metric("train_loss", avg_train, step=epoch)

        # —— 验证 —— #
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                preds = model(users, items)
                total_val += criterion(preds, ratings).item() * ratings.size(0)

        avg_val = total_val / len(val_loader.dataset)
        mlflow.log_metric("val_loss", avg_val, step=epoch)

        print(f"Epoch {epoch+1} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

    return model

def load_and_split(ratings_csv, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(ratings_csv)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 连续编码 userId/itemId
    df["user_idx"], _ = pd.factorize(df["userId"])
    df["item_idx"], _ = pd.factorize(df["movieId"])

    n = len(df)
    nt = int(n * (1 - val_ratio - test_ratio))
    nv = int(n * val_ratio)

    train_df = df.iloc[:nt]
    val_df   = df.iloc[nt:nt+nv]
    test_df  = df.iloc[nt+nv:]

    return train_df, val_df, test_df, df["user_idx"].nunique(), df["item_idx"].nunique()

if __name__ == "__main__":
    # —— 配置 —— #
    data_dir   = os.environ.get("MOVIELENS_DATA_DIR", "/mnt/MovieLens")
    ratings_csv= os.path.join(data_dir, "ratings.csv")
    batch_size = 1024
    emb_size   = 64
    lr         = 1e-3
    epochs     = 5

    print("Loading & splitting data…")
    train_df, val_df, test_df, n_users, n_items = load_and_split(ratings_csv)
    print(f"#train={len(train_df)}, #val={len(val_df)}, #test={len(test_df)}")
    print(f"n_users={n_users}, n_items={n_items}")

    train_ds = CSVDataset(train_df["user_idx"], train_df["item_idx"], train_df["rating"])
    val_ds   = CSVDataset(val_df["user_idx"],   val_df["item_idx"],   val_df["rating"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mlflow.set_experiment("movielens_recommender")
    with mlflow.start_run():
        # —— 记录超参数 —— #
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("embedding_size", emb_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        # —— 构建 & 训练 —— #
        model = RecommenderNet(n_users, n_items, embedding_size=emb_size)
        model = train_model(model, train_loader, val_loader, epochs, lr, device)

        # —— 保存 & 记录模型 —— #
        torch.save(model.state_dict(), "recommender.pth")
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("✅ Training + MLflow logging complete.")