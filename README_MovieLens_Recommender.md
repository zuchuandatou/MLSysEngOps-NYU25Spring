# MovieLens-20M Transformer-Based Recommender System

This project implements a **personalized movie recommendation system** using a **Transformer-based model (SSEPT)** trained on a **processed subset of the MovieLens-20M dataset**.

---

## 🧠 Project Goals

- Build a deep learning-based movie recommender using implicit feedback.
- Integrate **MLflow** for training experiment tracking.
- Deploy training on **Chameleon Cloud** using Docker and Jupyter environments.
- Enable **model retraining** from a pretrained checkpoint (`SSE_PT_tf.pth`).
- Optimize training performance using **Mixed Precision (AMP)** and **data subsampling**.

---

## 📁 Dataset: MovieLens-20M (x16x32 NPZ Format)

- Source: [https://grouplens.org/datasets/movielens/20m](https://grouplens.org/datasets/movielens/20m)
- Format: `.npz` files containing implicit interaction data as `[user_id, item_id]` pairs
- Directory structure after preprocessing:

```
/mnt/MovieLens-20M/organized/
├── training/       # trainx16x32_*.npz
├── validation/     # test_0.npz ~ test_7.npz
└── evaluation/     # test_8.npz ~ test_15.npz
```

---

## 🏗️ Model: SSEPT (Sequential Self-supervised Embedding + Transformer)

- `user_embedding_layer`: Embeds user indices
- `item_embedding_layer`: Embeds item indices
- `TransformerEncoder`: Processes user representations
- `fc_out`: Outputs predicted item logits

Model definition is located in: `train_memfix_amp.py`

---

## 🔁 Retraining

- Pretrained model: `SSE_PT_tf.pth`
- When retraining:
  - Embedding layers are skipped if size mismatch
  - Transformer encoder weights are loaded
  - Model resumes from pretrained encoder weights

Use:

```bash
python3 train_memfix_amp.py --retrain
```

---

## ⚙️ Training Setup

- Trained using **PyTorch + MLflow**
- Mixed Precision (AMP) via `torch.cuda.amp`
- Training script variants:
  - `train.py`: Baseline
  - `train_memfix_amp.py`: AMP + retrain + MLflow + 1/10 dataset
  - `train_with_gpu_logging_1of10.py`: includes GPU monitoring logic

---

## 📈 MLflow Logging

Logged parameters:

- `lr`, `batch_size`, `embedding_dim`
- Train/Val loss per epoch
- Artifact: trained `.pth` model

---

## 🐳 Docker and Data Integration

- Dataset preparation via `docker-compose-data.yaml`
- Preprocessed files copied or streamed into `/mnt/MovieLens-20M/organized`
- Dataset is automatically partitioned into `training`, `validation`, `evaluation`

---

## 🚀 Optimization Notes

- Used only 1/10th of `.npz` files to reduce memory and runtime
- AMP enabled for speedup and reduced GPU memory
- Batch size adjusted (2048 → 1024 → 512) based on GPU constraints
- GradScaler used to stabilize mixed precision training

---

## 🧪 Validation

- Training and validation losses logged via MLflow
- Trained model saved to disk and logged as artifact
- Can be reloaded for evaluation or future retraining

---

## ✅ Summary

This pipeline demonstrates an efficient, scalable, and trackable approach to training a Transformer-based movie recommender on MovieLens-20M, adapted for Chameleon cloud environments and limited compute environments.

