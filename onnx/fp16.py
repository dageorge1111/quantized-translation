import onnx
from onnxconverter_common import float16

# Input and output model paths
input_model_path = "../models/marianmt_model_float32.onnx"
output_model_path = "../models/marianmt_model_fp16.onnx"

# Load the model
model = onnx.load(input_model_path)

# Convert to FP16
model_fp16 = float16.convert_float_to_float16(model)

# Save the FP16 model
onnx.save(model_fp16, output_model_path)

print(f"FP16 model saved at {output_model_path}")

