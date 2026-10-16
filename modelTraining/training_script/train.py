import os
import time
import copy
import random
from glob import glob
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

# ===================== Hyperparameters =====================
DATA_DIR = '/mnt/object'
BATCH_SIZE = 128
LR = 0.001
MAXLEN = 200
USER_EMB_DIM = 50
ITEM_EMB_DIM = 50
NUM_BLOCKS = 6
NUM_HEADS = 1
DROPOUT_RATE = 0.2
THRESHOLD_USER = 0.92
THRESHOLD_ITEM = 0.9
NUM_EPOCHS = 201
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Data Loading =====================
def load_dataset(root):
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test  = defaultdict(list)
    all_items = set()

    def _load(part, container):
        part_dir = os.path.join(root, part)
        if not os.path.isdir(part_dir):
            print(f"[WARN] {part_dir} does not exist, skipping")
            return
        for txt in glob(os.path.join(part_dir, '*.txt')):
            with open(txt, 'r') as fp:
                for line in fp:
                    u, i = line.strip().split()
                    u = int(u); i = int(i)
                    container[u].append(i)
                    all_items.add(i)

    _load('training',   user_train)
    _load('validation', user_valid)
    _load('evaluation', user_test)

    max_user = max(
        max(user_train.keys(), default=0),
        max(user_valid.keys(), default=0),
        max(user_test.keys(),  default=0)
    )
    max_item = max(all_items) if all_items else 0
    return user_train, user_valid, user_test, max_user, max_item

# ===================== Sampler =====================

def random_neq(left, right, s):
    t = random.randint(left, right-1)
    while t in s:
        t = random.randint(left, right-1)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen,
                    threshold_user, threshold_item, result_queue, seed):
    random.seed(seed)
    np.random.seed(seed)
    users = list(user_train.keys())
    while True:
        batch = []
        for _ in range(batch_size):
            u = random.choice(users)
            seq = np.zeros(maxlen, dtype=np.int32)
            pos = np.zeros(maxlen, dtype=np.int32)
            neg = np.zeros(maxlen, dtype=np.int32)
            history = user_train[u]
            if not history: history = [0]
            nxt = history[-1]
            idx = maxlen - 1
            ts = set(history)
            for item in reversed(history[:-1]):
                seq[idx] = item
                # SSE on item side
                if random.random() > threshold_item:
                    item = random.randint(1, itemnum)
                    nxt  = random.randint(1, itemnum)
                pos[idx] = nxt
                if nxt!=0:
                    neg[idx] = random_neq(1, itemnum+1, ts)
                nxt = item
                idx -= 1
                if idx<0: break
            # SSE on user side
            if random.random() > threshold_user:
                u = random.randint(1, usernum)
            batch.append((u, seq, pos, neg))
        # transpose batch
        users_b, seq_b, pos_b, neg_b = zip(*batch)
        result_queue.put((np.array(users_b), np.array(seq_b), np.array(pos_b), np.array(neg_b)))


class WarpSampler:
    def __init__(self, user_train, usernum, itemnum,
                 batch_size, maxlen, threshold_user, threshold_item, n_workers=3):
        self.result_queue = Queue(maxsize=n_workers*10)
        self.processes = []
        for _ in range(n_workers):
            p = Process(target=sample_function,
                        args=(user_train, usernum, itemnum,
                              batch_size, maxlen,
                              threshold_user, threshold_item,
                              self.result_queue, random.randint(0,1e9)))
            p.daemon = True
            p.start()
            self.processes.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processes:
            p.terminate()
            p.join()

# ===================== Evaluation =====================

def evaluate_valid(model, dataset):
    user_train, user_valid, _, usernum, itemnum = dataset
    NDCG = HT = cnt = 0
    users = list(user_valid.keys())
    for u in users:
        if not user_train[u] or not user_valid[u]: continue
        seq = np.zeros(MAXLEN, dtype=np.int32)
        idx = MAXLEN-1
        for it in reversed(user_train[u]):
            seq[idx] = it; idx-=1
            if idx<0: break
        rated = set(user_train[u]) | {0}
        target = user_valid[u][0]
        candidates = [target]
        while len(candidates)<101:
            t = random.randint(1, itemnum)
            if t not in rated: candidates.append(t)
        u_t = torch.LongTensor([u]).to(DEVICE)
        seq_t = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
        items_t = torch.LongTensor(candidates).unsqueeze(0).to(DEVICE)
        preds = -model.predict(u_t, seq_t, items_t)[0].cpu().numpy()
        rank = preds.argsort().argsort()[0]
        cnt+=1
        if rank<10:
            HT +=1; NDCG += 1/np.log2(rank+2)
    return NDCG/cnt, HT/cnt


def evaluate(model, dataset):
    user_train, _, user_test, usernum, itemnum = dataset
    NDCG = HT = cnt = 0
    users = list(user_test.keys())
    for u in users:
        if not user_train[u] or not user_test[u]: continue
        seq = np.zeros(MAXLEN, dtype=np.int32)
        idx = MAXLEN-1
        for it in reversed(user_train[u]):
            seq[idx] = it; idx-=1
            if idx<0: break
        rated = set(user_train[u]) | {0}
        target = user_test[u][0]
        candidates = [target]
        while len(candidates)<101:
            t = random.randint(1, itemnum)
            if t not in rated: candidates.append(t)
        u_t = torch.LongTensor([u]).to(DEVICE)
        seq_t = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
        items_t = torch.LongTensor(candidates).unsqueeze(0).to(DEVICE)
        preds = -model.predict(u_t, seq_t, items_t)[0].cpu().numpy()
        rank = preds.argsort().argsort()[0]
        cnt+=1
        if rank<10:
            HT +=1; NDCG += 1/np.log2(rank+2)
    return NDCG/cnt, HT/cnt

# ===================== Model Definition =====================
class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(1,2))))))
        y = y.transpose(1,2)
        return x + y

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = PointWiseFeedForward(dim, dropout)
        self.ln  = nn.LayerNorm(dim)

    def forward(self, x):
        mask = ~torch.tril(torch.ones((x.size(1), x.size(1)), dtype=torch.bool, device=x.device))
        y,_ = self.mha(self.ln(x), x, x, attn_mask=mask)
        x = x + y
        x = self.ffn(self.ln(x))
        return x

class SSEPT(nn.Module):
    def __init__(self, user_count, item_count, **kw):
        super().__init__()
        self.seq_len = MAXLEN
        self.hidden  = ITEM_EMB_DIM + USER_EMB_DIM
        self.item_emb = nn.Embedding(item_count+1, ITEM_EMB_DIM, padding_idx=0)
        self.user_emb = nn.Embedding(user_count+1, USER_EMB_DIM, padding_idx=0)
        self.pos_emb  = nn.Embedding(self.seq_len, self.hidden)
        self.layers   = nn.ModuleList([TransformerBlock(self.hidden, NUM_HEADS, DROPOUT_RATE)
                                        for _ in range(NUM_BLOCKS)])
        self.ln_final = nn.LayerNorm(self.hidden)

    def embed(self, u, seq):
        ie = self.item_emb(seq) * np.sqrt(ITEM_EMB_DIM)
        ue = self.user_emb(u).unsqueeze(1) * np.sqrt(USER_EMB_DIM)
        ue = ue.expand(-1, seq.size(1), -1)
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        pe  = self.pos_emb(pos)
        x = torch.cat([ie, ue], -1) + pe
        mask = seq==0
        return x * (~mask.unsqueeze(-1)), ue

    def forward(self, u, seq, pos, neg):
        x, ue = self.embed(u, seq)
        for l in self.layers: x = l(x)
        feats = self.ln_final(x)
        pe = self.item_emb(pos)
        ne = self.item_emb(neg)
        pe = torch.cat([pe, ue], -1)
        ne = torch.cat([ne, ue], -1)
        return (feats*pe).sum(-1), (feats*ne).sum(-1)

    def predict(self, u, seq, items):
        x, ue = self.embed(u, seq)
        for l in self.layers: x = l(x)
        feat = self.ln_final(x)[:, -1]
        ie = self.item_emb(items)
        ue_expand = ue[:, :items.size(1), :]
        emb = torch.cat([ie, ue_expand], -1)
        return emb.matmul(feat.unsqueeze(-1)).squeeze(-1)

# ===================== Training Loop =====================
def main():
    # load
    user_train, user_valid, user_test, usernum, itemnum = load_dataset(DATA_DIR)
    dataset = (user_train, user_valid, user_test, usernum, itemnum)
    num_batch = len(user_train) // BATCH_SIZE

    # model
    model = SSEPT(usernum, itemnum).to(DEVICE)
    for p in model.parameters():
        if p.dim()>1: nn.init.xavier_normal_(p)

    # mlflow
    mlflow.set_experiment("SSEPT_ML32M")
    with mlflow.start_run():
        mlflow.log_params({
            'lr': LR, 'batch_size': BATCH_SIZE,
            'maxlen': MAXLEN, 'num_blocks': NUM_BLOCKS,
            'user_emb_dim': USER_EMB_DIM, 'item_emb_dim': ITEM_EMB_DIM
        })

        sampler = WarpSampler(user_train, usernum, itemnum,
                              BATCH_SIZE, MAXLEN,
                              THRESHOLD_USER, THRESHOLD_ITEM, n_workers=3)
        criterion = nn.BCEWithLogitsLoss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        t0 = time.time(); T=0
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss=0
            for _ in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u  = torch.LongTensor(u).to(DEVICE)
                seq= torch.LongTensor(seq).to(DEVICE)
                pos= torch.LongTensor(pos).to(DEVICE)
                neg= torch.LongTensor(neg).to(DEVICE)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                optimizer.zero_grad()
                idx = (pos.cpu().numpy()!=0).nonzero()
                loss = criterion(pos_logits[idx], pos_labels[idx]) + criterion(neg_logits[idx], neg_labels[idx])
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / num_batch

            if epoch % 20 == 0:
                model.eval()
                t1 = time.time() - t0; T += t1; t0 = time.time()
                v_ndcg, v_ht = evaluate_valid(model, dataset)
                t_ndcg, t_ht = evaluate(model, dataset)
                mlflow.log_metrics({
                    'train_loss': avg_loss,
                    'val_ndcg': v_ndcg, 'val_ht': v_ht,
                    'test_ndcg': t_ndcg, 'test_ht': t_ht
                }, step=epoch)
                print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | V(NDCG@10 {v_ndcg:.4f}, HT@10 {v_ht:.4f}) | T(NDCG@10 {t_ndcg:.4f}, HT@10 {t_ht:.4f})")

        # save
        ckpt = 'ssept_ml32m.pth'
        torch.save(model.state_dict(), ckpt)
        mlflow.log_artifact(ckpt)

if __name__ == '__main__':
    main()
