
import onnx
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

# Path to ONNX model
input_model_path = "model/ssept.onnx"
output_model_path = "model/ssept_optimized.onnx"

# Load and check ONNX model
model = onnx.load(input_model_path)
onnx.checker.check_model(model)
print("Loaded and checked model successfully.")


# Create session with optimization enabled
opt = SessionOptions()
opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

# If using GPU later, switch to CUDAExecutionProvider or TensorrtExecutionProvider
sess = InferenceSession(input_model_path, sess_options=opt, providers=["CPUExecutionProvider"])

# Save optimized model
optimized_model_bytes = sess.serialize()
with open(output_model_path, "wb") as f:
    f.write(optimized_model_bytes)
print(f"Optimized model saved to {output_model_path}")
