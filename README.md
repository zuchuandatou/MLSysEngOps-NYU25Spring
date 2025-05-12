# SSEPT Model Serving

## Model-Level optimization

This directory contains scripts and configuration files for serving and benchmarking the SSEPT recommendation model. It supports both PyTorch-based and ONNX-optimized inference pipelines, and includes model quantization and execution provider tuning.

### Directory Structure

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
│   ├── Dockerfile.jupyter-onnx-gpu     # ONNX runtime Jupyter environment
│   ├── Dockerfile.jupyter-onnx-cpu     # CPU-only ONNX runtime Jupyter environment
│   └── docker-compose-data.yaml        # Compose config for launching container
```

### Launch Jupyter Benchmark Container

To evaluate model latency and throughput in a CPU environment:

```bash
docker compose -f docker/docker-compose-data.yaml up -d
```

You can access the Jupyter interface at `http://<your-node-ip>:8888`.

### Export PyTorch to ONNX

We provide export scripts using `torch.onnx.export` with dynamic axes, to allow variable-length batches and flexible batch inference.

### Model Optimizations

Implemented and benchmarked optimization strategies:

- **Graph Optimization** (ONNX Runtime)
- **Dynamic Quantization** (Intel Neural Compressor)
- **Static Quantization** with calibration set
- **Execution Provider Tuning** (CPU, OpenVINO, TensorRT)

### FastAPI Model Serving

To launch an HTTP API using the trained model:

```bash
cd workspace/models
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Evaluation Metrics

Each notebook reports:

- Accuracy
- Inference latency (median, p95, p99)
- Inference FPS


##Notes

- Set `MOVIELENS_DATA_DIR=/mnt/data` to locate validation/test CSVs inside container
- ONNX Runtime EP choice impacts performance significantly (CPU vs OpenVINO vs TensorRT)、



## System-Level Optimization 

This section focuses on evaluating the inference performance of our ONNX models on low-resource edge devices, such as Jetson Nano or CPU-only microservers.

We benchmarked the following model variants:

- **Baseline FP32 ONNX model**
- **Dynamically quantized ONNX model** (`quant_dynamic`)
- **Statically quantized ONNX model** (`quant_static_aggressive`)

All benchmarks are run using **synthetic dummy inputs**, simulating real traffic but avoiding the overhead of loading actual test data. This makes it ideal for edge performance profiling.

### Benchmark Setup

We use `onnxruntime` with `CPUExecutionProvider` and the following configuration:

```python
dummy_user = np.random.randint(0, 1000, size=(1,), dtype=np.int64)
dummy_seq = np.random.randint(0, 1000, size=(1, SEQ_LEN), dtype=np.int64)

ort_session.run(None, {
    "user": dummy_user,
    "sequence": dummy_seq
})
```

Each model is tested for:

- **Inference latency** (median, 95th, 99th percentile)
- **Single-sample throughput (FPS)**

### Observations

- Dynamic quantization reduced model size and offered modest speed-up on CPU.
- Static quantization (aggressive) led to slightly higher latency due to quantize/dequantize ops, but still showed storage benefits.
- All models retained prediction accuracy when compared using real validation data.

### Evaluated Models

| Model Variant                   | Path                                                 |
|--------------------------------|------------------------------------------------------|
| Baseline FP32                  | `models/SSE_PT10kemb.onnx`                           |
| Dynamic Quantized              | `models/SSE_PT10kemb_quant_dynamic.onnx`            |
| Static Quantized (Aggressive) | `models/SSE_PT10kemb_quant_static_aggressive.onnx`  |

### Recommended Use

- Use **quant_dynamic** on general CPU platforms with minimal setup.
- Use **quant_static_aggressive** if you can afford calibration and want to maximize storage/compute efficiency.
- Consider compiling with **OpenVINOExecutionProvider** or **TensorRTExecutionProvider** for best edge performance.


## Serving SSEPT Model with Triton ONNX Backend

This section describes deploying the ONNX version of the SSEPT recommendation model using Triton Inference Server for optimized, production-grade inference performance.

---

### Run with Docker Compose

To deploy the ONNX backend in Triton:

```bash
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

Check status:

```bash
docker logs triton_server
```

should see `recommender_model_onnx` is loaded and marked `READY`.

---

### Benchmark with `perf_analyzer`

From within the Jupyter container:

```bash
perf_analyzer -u triton_server:8000 -m recommender_model_onnx \
  --shape USER_ID:1 SEQ:50 -b 1 --concurrency-range 8
```

To increase concurrency:

```bash
perf_analyzer -u triton_server:8000 -m recommender_model_onnx \
  --shape USER_ID:1 SEQ:50 -b 1 --concurrency-range 16
```

---

### Enable Multi-Instance Deployment

To scale across GPUs, update `instance_group`:

```protobuf
instance_group [
  { count: 2 kind: KIND_GPU gpus: [0] },
  { count: 2 kind: KIND_GPU gpus: [1] }
]
```

Then re-build and restart:

```bash
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml build triton_server
docker compose -f ~/serve-system-chi/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

---

### Summary

- Use `recommender_model_onnx` as your model name
- Match input/output tensor names with ONNX file
- Use `perf_analyzer` to verify batching and latency
- Use `nvtop` or `nvidia-smi` to monitor GPU utilization

