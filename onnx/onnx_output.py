import onnxruntime as ort

# Load your ONNX model
onnx_model_path = "marianmt_quantized.onnx"
session = ort.InferenceSession(onnx_model_path)

# Print available input and output names in the model
print("Input names:", [input.name for input in session.get_inputs()])
print("Output names:", [output.name for output in session.get_outputs()])

