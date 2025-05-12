
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import RecommenderModel  # 假设你的模型类名为 RecommenderModel
from dataset import RatingDataset  # 假设你有这个数据集类
from tqdm import tqdm

# 启用 cudnn benchmark
torch.backends.cudnn.benchmark = True

# 超参数
BATCH_SIZE = 1024
EPOCHS = 1
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train_npz = "/mnt/MovieLens-20M/organized/training/trainx16x32_0.npz"
val_npz = "/mnt/MovieLens-20M/organized/validation/testx16x32_0.npz"

train_data = RatingDataset(train_npz)
val_data = RatingDataset(val_npz)

# 仅用四分之一数据用于快速测试
train_data = train_data[:len(train_data)//4]
val_data = val_data[:len(val_data)//4]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# 获取用户和物品总数
num_users = max(train_data.users.max().item(), val_data.users.max().item()) + 1
num_items = max(train_data.items.max().item(), val_data.items.max().item()) + 1

# 初始化模型
model = RecommenderModel(num_users=num_users, num_items=num_items).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

# 训练过程
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        users, items = batch[:, 0].to(DEVICE, non_blocking=True), batch[:, 1].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(users, items)
            labels = torch.ones_like(preds).to(DEVICE)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1))
