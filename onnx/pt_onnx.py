from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the pre-trained MarianMT model
model_name = 'Helsinki-NLP/opus-mt-en-de'  # Replace with your desired model
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# Set the model to evaluation mode
model.eval()

# Example input
input_text = "Hello world!"
inputs = tokenizer([input_text], return_tensors="pt")

# Create dummy decoder_input_ids for export
decoder_start_token_id = model.config.decoder_start_token_id
batch_size, seq_length = inputs['input_ids'].shape
decoder_input_ids = torch.full(
    (batch_size, 1), decoder_start_token_id, dtype=torch.long
)

# Export the model to ONNX
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask'], decoder_input_ids),
    "../models/marianmt_model.onnx",
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_length"}
    },
    opset_version=11
)

