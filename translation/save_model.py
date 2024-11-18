import torch
import onnx
from pathlib import Path
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

from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths for the original and quantized ONNX models
onnx_model_path = "../models/marianmt_model_float32.onnx"
quantized_model_path = "../models/marianmt_model_int8.onnx"

# Perform dynamic quantization to INT8
quantize_dynamic(
    model_input=onnx_model_path,                # Input model path
    model_output=quantized_model_path,          # Output model path
    weight_type=QuantType.QInt8                # Quantize weights to INT8
)

print(f"Quantized model saved at '{quantized_model_path}'.")


print(f"ONNX model saved at '{onnx_model_path}'.")
print(f"Tokenizer directory: {tokenizer_dir}")

