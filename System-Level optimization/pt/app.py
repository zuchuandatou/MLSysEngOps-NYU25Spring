"""
Minimal FastAPI API for the SSEPT ONNX recommendation model
--------------------------------------------------------------
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
POST: curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
        -d '{"user_id": 42, "sequence": [10,11,23,99], "top_k": 5}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import numpy as np
import onnxruntime as ort
import traceback
import os

app = FastAPI(title="SSEPT ONNX Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    user_id: int
    sequence: List[int]
    top_k: int = 1

class PredictResponse(BaseModel):
    top_items: List[int]

class TestResponse(BaseModel):
    message: str

class VersionResponse(BaseModel):
    model_version: str

SEQ_LEN = 100
MODEL_PATH = "models/SSE_PT10kemb_quant_dynamic.onnx"

try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f" Model loaded from {MODEL_PATH}")
    input_names = [i.name for i in ort_session.get_inputs()]
    print(f" ONNX model input names: {input_names}")
except Exception as e:
    traceback.print_exc()
    raise RuntimeError("Failed to load ONNX model")


def pad_or_truncate(seq, max_len):
    return seq[-max_len:] + [0] * (max_len - len(seq))


@app.get("/test", response_model=TestResponse)
@app.post("/test", response_model=TestResponse)
async def test_endpoint():
    return TestResponse(message="CORS is working! Your API is accessible from the frontend.")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        seq_padded = pad_or_truncate(request.sequence, SEQ_LEN)
        user_ids = np.array([[request.user_id]], dtype=np.int64)
        item_seqs = np.array([seq_padded], dtype=np.int64)

        outputs = ort_session.run(None, {
            "user_ids": user_ids,
            "item_seqs": item_seqs
        })

        logits = outputs[0]
        top_k_indices = np.argsort(logits[0])[::-1][:request.top_k]
        return PredictResponse(top_items=top_k_indices.tolist())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

VERSION_FILE = Path(os.getenv("VERSION_FILE", "versions.txt"))

@app.get("/version", response_model=VersionResponse)
async def get_version():
    if not VERSION_FILE.exists():
        raise HTTPException(status_code=404, detail="versions.txt not found")
    return VersionResponse(model_version=VERSION_FILE.read_text().strip())

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
