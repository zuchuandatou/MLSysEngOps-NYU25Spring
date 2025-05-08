"""
Minimal FastAPI API for the SSEPT sequential‑recommendation model
--------------------------------------------------------------
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
POST: curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
        -d '{"user_id": 42, "sequence": [10,11,23,99], "top_k": 5}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import traceback
import os
import uvicorn

# Import from utilities
from utilities import build_model_from_ckpt, pad_or_truncate

# Define Pydantic models for request/response validation
class PredictRequest(BaseModel):
    user_id: int
    sequence: List[int]
    top_k: int = 1

class PredictResponse(BaseModel):
    top_items: List[int]

class TestResponse(BaseModel):
    message: str

# Create FastAPI app
app = FastAPI(title="SSEPT Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
# Change this to your checkpoint path or set env var MODEL_PATH
MODEL_PATH = "SSE_PT10kemb.pth"

# Model initialization
try:
    model = build_model_from_ckpt(MODEL_PATH, DEVICE)
    SEQ_LEN = model.seq_max_len
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Checkpoint file '{MODEL_PATH}' not found.")
    raise RuntimeError(
        f"Checkpoint '{MODEL_PATH}' not found. "
        "Set MODEL_PATH env var or place the .pth file next to app.py."
    )
except ValueError as e:
    print(f"Error loading checkpoint: {e}")
    raise RuntimeError(f"Failed to load checkpoint: {e}")
except Exception as e:
    print(f"Unexpected error loading model: {e}")
    traceback.print_exc()
    raise RuntimeError(f"Failed to initialize model: {e}")


@app.get("/test", response_model=TestResponse)
@app.post("/test", response_model=TestResponse)
async def test_endpoint():
    """Simple endpoint to test if CORS is working properly"""
    print("Test endpoint called")
    return TestResponse(message="CORS is working! Your API is accessible from the frontend.")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict top items for a user based on their interaction sequence
    """
    print(f"Processing request: user_id={request.user_id}, sequence={request.sequence}, top_k={request.top_k}")

    try:
        # Tensor preparation
        seq_padded = pad_or_truncate(request.sequence, SEQ_LEN)
        user_tensor = torch.tensor([request.user_id], dtype=torch.long, device=DEVICE)
        seq_tensor = torch.tensor([seq_padded], dtype=torch.long, device=DEVICE)

        # Model inference
        with torch.no_grad():
            logits = model(user_tensor, seq_tensor)  # [1, item_num]
            _, top_indices = torch.topk(logits, k=request.top_k, dim=1)
            top_items = top_indices.squeeze(0).cpu().tolist()
        
        print(f"Returning top items: {top_items}")
        return PredictResponse(top_items=top_items)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    # Start the server using Uvicorn
    print("Starting FastAPI server...")
    print("* CORS enabled for all origins (testing mode)")
    print("* API docs: http://127.0.0.1:8000/docs")
    print("* Test endpoint: http://127.0.0.1:8000/test")
    print("* Predict endpoint: http://127.0.0.1:8000/predict (POST)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
