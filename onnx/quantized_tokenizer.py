import torch
import numpy as np
import onnx
from pathlib import Path
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from transformers import MarianMTModel, MarianTokenizer, MarianConfig

# Define the model name
model_name = 'Helsinki-NLP/opus-mt-en-de'

# Load the MarianMT PyTorch model and ensure it’s in float32
model = MarianMTModel.from_pretrained(model_name)
model = model.float()  # Convert model to float32

# Confirm all parameters are in float32
for param in model.parameters():
    assert param.dtype == torch.float32, "Model parameter is not in float32"

# Initialize tokenizer and configuration
tokenizer = MarianTokenizer.from_pretrained(model_name)
config = MarianConfig.from_pretrained(model_name)

# Save tokenizer locally
tokenizer_dir = Path("../models/tokenizer")
tokenizer.save_pretrained(tokenizer_dir)
print(f"Tokenizer saved at '{tokenizer_dir}'.")

# Create dummy inputs for ONNX export
input_text = "This is a test sentence."
dummy_inputs = tokenizer(input_text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
input_ids = dummy_inputs["input_ids"]
attention_mask = dummy_inputs["attention_mask"]

# Add dummy decoder_input_ids
decoder_start_token_id = config.decoder_start_token_id
decoder_input_ids = torch.full((input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long)

# Export the model to ONNX in float32 precision
onnx_model_path = "../models/marianmt_model_float32.onnx"
torch.onnx.export(
    model,
    (input_ids, attention_mask, decoder_input_ids),
    onnx_model_path,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size", 1: "seq_length"}
    }
)

print("ONNX model exported with float32 precision.")

# Verify all tensors in the ONNX model are float32
onnx_model = onnx.load(onnx_model_path)
for initializer in onnx_model.graph.initializer:
    assert initializer.data_type == onnx.TensorProto.FLOAT, f"{initializer.name} is not float32."
print("All model parameters in ONNX model are confirmed to be float32.")

# Define calibration data for quantization
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_inputs):
        self.data = calibration_inputs
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

# Prepare a small set of calibration data
calibration_inputs = []
sample_texts = [
    "hello, how are you",
    "what time is it",
    "the capital of france is paris."
]

for text in sample_texts:
    inputs = tokenizer([text], return_tensors="np", max_length=64, truncation=True, padding="max_length")
    batch_size = inputs['input_ids'].shape[0]
    decoder_input_ids = np.full((batch_size, 1), config.decoder_start_token_id, dtype=np.int64)
    calibration_inputs.append({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'decoder_input_ids': decoder_input_ids
    })

# Perform static quantization
quantized_model_path = "../models/marianmt_quantized_static_tokenize.onnx"
quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=MyCalibrationDataReader(calibration_inputs),
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print(f"Quantized model saved as '{quantized_model_path}'.")
print(f"Tokenizer directory: {tokenizer_dir}")

