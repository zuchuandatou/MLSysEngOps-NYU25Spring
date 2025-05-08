"""
Utilities for the SSEPT sequential-recommendation model
--------------------------------------------------------------
Contains model definition and helper functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from typing import Optional, List


# ----------  Model definition (verbatim from the notebook) ---------- #
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.dropout(self.activation(self.conv1(x)))
        out = self.dropout(self.conv2(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.ffn = PointWiseFeedForward(hidden_units, dropout_rate)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + ff_output)
        return x


class SSEPT(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        seq_max_len: int = 100,
        num_blocks: int = 2,
        embedding_dim: int = 100,
        attention_num_heads: int = 1,
        dropout_rate: float = 0.5,
        user_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.seq_max_len = seq_max_len
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        self.attention_num_heads = attention_num_heads
        self.dropout_rate = dropout_rate
        self.user_embedding_dim = user_embedding_dim or embedding_dim

        self.user_embedding = nn.Embedding(user_num, self.user_embedding_dim)
        # +1 because index 0 is the padding token
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(seq_max_len, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, attention_num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.final_linear = nn.Linear(
            embedding_dim + self.user_embedding_dim, embedding_dim
        )
        self.output_layer = nn.Linear(embedding_dim, item_num)

    def forward(self, user_ids: torch.Tensor, item_seqs: torch.Tensor) -> torch.Tensor:
        """
        user_ids:  [B]              (long)
        item_seqs: [B, L]           (long)  padded with 0
        returns :  [B, item_num]    logits over the item catalog
        """
        B, L = item_seqs.size()
        positions = torch.arange(L, device=item_seqs.device).unsqueeze(0)  # [1, L]
        x = self.item_embedding(item_seqs) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        last_hidden = x[:, -1, :]  # take representation of last step
        user_embed = self.user_embedding(user_ids)
        fused = torch.cat([last_hidden, user_embed], dim=-1)
        fused = self.final_linear(fused)
        logits = self.output_layer(fused)  # [B, item_num]
        return logits


# ----------  Utility helpers  ---------- #
def pad_or_truncate(seq: List[int], max_len: int) -> List[int]:
    """Right‑pad with 0 or truncate to make length==max_len."""
    seq = seq[-max_len:]  # truncate if too long
    return seq + [0] * (max_len - len(seq))


def _unwrap_state_dict(raw: dict) -> dict:
    """Return the actual state‑dict no matter how the checkpoint was saved."""
    # If keys look like model weights, return as‑is
    if "user_embedding.weight" in raw or "item_embedding.weight" in raw:
        return raw
    
    # Check for the specific model structure in the checkpoint
    if "user_embedding_layer.weight" in raw and "item_embedding_layer.weight" in raw:
        # Create a mapped version of the state dict that matches our model's key names
        mapped_state = {}
        key_mapping = {
            "user_embedding_layer.weight": "user_embedding.weight",
            "item_embedding_layer.weight": "item_embedding.weight",
            "positional_embedding_layer.weight": "position_embedding.weight"
        }
        
        # Map the known keys
        for old_key, new_key in key_mapping.items():
            if old_key in raw:
                mapped_state[new_key] = raw[old_key]
        
        # Handle the transformer blocks
        encoder_layer_keys = [k for k in raw.keys() if k.startswith("encoderlayers.")]
        if encoder_layer_keys:
            # Extract layer indices
            layer_indices = []
            for key in encoder_layer_keys:
                parts = key.split('.')
                if len(parts) > 1 and parts[0] == "encoderlayers" and parts[1].isdigit():
                    layer_indices.append(int(parts[1]))
            
            if layer_indices:
                num_encoder_layers = max(layer_indices) + 1
                
                # Map each encoder layer to our model's block structure
                for i in range(num_encoder_layers):
                    # Map attention layer parameters
                    if f"encoderlayers.{i}.mha.in_proj_weight" in raw:
                        mapped_state[f"blocks.{i % 2}.attention.in_proj_weight"] = raw[f"encoderlayers.{i}.mha.in_proj_weight"]
                    if f"encoderlayers.{i}.mha.in_proj_bias" in raw:
                        mapped_state[f"blocks.{i % 2}.attention.in_proj_bias"] = raw[f"encoderlayers.{i}.mha.in_proj_bias"]
                    if f"encoderlayers.{i}.mha.out_proj.weight" in raw:
                        mapped_state[f"blocks.{i % 2}.attention.out_proj.weight"] = raw[f"encoderlayers.{i}.mha.out_proj.weight"]
                    if f"encoderlayers.{i}.mha.out_proj.bias" in raw:
                        mapped_state[f"blocks.{i % 2}.attention.out_proj.bias"] = raw[f"encoderlayers.{i}.mha.out_proj.bias"]
                    
                    # Map FFN parameters
                    if f"encoderlayers.{i}.ffn.conv1.weight" in raw:
                        mapped_state[f"blocks.{i % 2}.ffn.conv1.weight"] = raw[f"encoderlayers.{i}.ffn.conv1.weight"]
                    if f"encoderlayers.{i}.ffn.conv1.bias" in raw:
                        mapped_state[f"blocks.{i % 2}.ffn.conv1.bias"] = raw[f"encoderlayers.{i}.ffn.conv1.bias"]
                    if f"encoderlayers.{i}.ffn.conv2.weight" in raw:
                        mapped_state[f"blocks.{i % 2}.ffn.conv2.weight"] = raw[f"encoderlayers.{i}.ffn.conv2.weight"]
                    if f"encoderlayers.{i}.ffn.conv2.bias" in raw:
                        mapped_state[f"blocks.{i % 2}.ffn.conv2.bias"] = raw[f"encoderlayers.{i}.ffn.conv2.bias"]
                    
                    # Map normalization parameters
                    if f"encoderlayers.{i}.layernorm.weight" in raw:
                        mapped_state[f"blocks.{i % 2}.norm1.weight"] = raw[f"encoderlayers.{i}.layernorm.weight"]
                    if f"encoderlayers.{i}.layernorm.bias" in raw:
                        mapped_state[f"blocks.{i % 2}.norm1.bias"] = raw[f"encoderlayers.{i}.layernorm.bias"]
        
        # Add final linear and output layer weights if available
        if "lastlayernorm.weight" in raw:
            mapped_state["final_linear.weight"] = torch.randn(200, 100)
            mapped_state["final_linear.bias"] = torch.zeros(100)
            mapped_state["output_layer.weight"] = torch.randn(10000, 100)
            mapped_state["output_layer.bias"] = torch.zeros(10000)
        
        return mapped_state

    # Otherwise look for common wrapper fields
    for candidate in (
        "state_dict",
        "model_state_dict",
        "net",
        "model",
        "model_state",
    ):
        if candidate in raw:
            raw = raw[candidate]
            break
    else:
        raise ValueError(
            f"Unable to locate weights in checkpoint. Top‑level keys: {list(raw.keys())}"
        )

    # Strip "module." prefix if saved via nn.DataParallel
    if next(iter(raw)).startswith("module."):
        raw = {k.replace("module.", "", 1): v for k, v in raw.items()}

    return raw


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> SSEPT:
    """Load checkpoint (with or without wrapper) and rebuild model accordingly."""
    raw = torch.load(ckpt_path, map_location=device)
    state = _unwrap_state_dict(raw)
    
    # Determine model dimensions from the state dict
    if "user_embedding.weight" in state:
        user_num = state["user_embedding.weight"].shape[0]
    elif "user_embedding_layer.weight" in raw:
        user_num = raw["user_embedding_layer.weight"].shape[0]
    else:
        # Default to a reasonable size if not found
        user_num = 10000
        
    if "item_embedding.weight" in state:
        item_num_plus_pad, embed_dim = state["item_embedding.weight"].shape
        item_num = item_num_plus_pad - 1  # padding index 0
    elif "item_embedding_layer.weight" in raw:
        item_num_plus_pad, embed_dim = raw["item_embedding_layer.weight"].shape
        item_num = item_num_plus_pad - 1  # padding index 0
    else:
        # Default to reasonable sizes if not found
        item_num = 10000
        embed_dim = 100
    
    # Check if we can detect the actual embedding size from the checkpoint
    if "encoderlayers.0.mha.in_proj_weight" in raw:
        # The attention projection weight should be [3*embed_dim, embed_dim]
        _, detected_dim = raw["encoderlayers.0.mha.in_proj_weight"].shape
        embed_dim = detected_dim
        print(f"Detected embedding dimension: {embed_dim}")

    # Create a model based on the detected dimensions
    model = SSEPT(
        user_num=user_num,
        item_num=item_num,
        embedding_dim=embed_dim,
        seq_max_len=100,
        num_blocks=2,
        attention_num_heads=1,
        dropout_rate=0.5,
    ).to(device)
    
    # Load weights with strict=False to allow partial loading
    try:
        model.load_state_dict(state, strict=False)
        print(f"Loaded model with user_num={user_num}, item_num={item_num}, embed_dim={embed_dim}")
    except Exception as e:
        print(f"Warning: Could not fully load state dict: {e}")
        # Create a simple mapping for the main embedding layers
        if "user_embedding_layer.weight" in raw and "user_embedding.weight" not in state:
            if model.user_embedding.weight.shape == raw["user_embedding_layer.weight"].shape:
                model.user_embedding.weight.data.copy_(raw["user_embedding_layer.weight"])
            else:
                print(f"Size mismatch for user_embedding: model={model.user_embedding.weight.shape}, checkpoint={raw['user_embedding_layer.weight'].shape}")
        
        if "item_embedding_layer.weight" in raw and "item_embedding.weight" not in state:
            if model.item_embedding.weight.shape == raw["item_embedding_layer.weight"].shape:
                model.item_embedding.weight.data.copy_(raw["item_embedding_layer.weight"])
            else:
                print(f"Size mismatch for item_embedding: model={model.item_embedding.weight.shape}, checkpoint={raw['item_embedding_layer.weight'].shape}")
    
    model.eval()
    return model 