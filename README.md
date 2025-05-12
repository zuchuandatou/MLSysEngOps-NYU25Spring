# SSEPT Model Serving

This directory contains scripts and configuration files for serving and benchmarking the SSEPT recommendation model. It supports both PyTorch-based and ONNX-optimized inference pipelines, and includes model quantization and execution provider tuning.

## Directory Structure

```
serve-model-chi/
│
├── workspace/
│   ├── models/
│   │   ├── SSE_PT10kemb.pth            # Trained PyTorch model
│   │   ├── *.onnx                      # Exported ONNX models
│   ├── utilities.py                    # Helper for building and calling the model
│   ├── app.py                          # FastAPI server for online inference
│   └── benchmark_*.ipynb               # Offline benchmark notebooks
│
├── docker/
│   ├── Dockerfile.jupyter-onnx-cpu     # CPU-only ONNX runtime Jupyter environment
│   └── docker-compose-data.yaml        # Compose config for launching container
```

## Launch Jupyter Benchmark Container

To evaluate model latency and throughput in a CPU environment:

```bash
docker compose -f docker/docker-compose-data.yaml up -d
```

You can access the Jupyter interface at `http://<your-node-ip>:8888`.

## Export PyTorch to ONNX

We provide export scripts using `torch.onnx.export` with dynamic axes, to allow variable-length batches and flexible batch inference.

## Model Optimizations

Implemented and benchmarked optimization strategies:

- **Graph Optimization** (ONNX Runtime)
- **Dynamic Quantization** (Intel Neural Compressor)
- **Static Quantization** with calibration set
- **Execution Provider Tuning** (CPU, OpenVINO, TensorRT)

## FastAPI Model Serving

To launch an HTTP API using the trained model:

```bash
cd workspace/models
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Evaluation Metrics

Each notebook reports:

- Accuracy
- Inference latency (median, p95, p99)
- Inference FPS


##Notes

- Set `MOVIELENS_DATA_DIR=/mnt/data` to locate validation/test CSVs inside container
- ONNX Runtime EP choice impacts performance significantly (CPU vs OpenVINO vs TensorRT)
