import torch
from transformers import MarianMTModel

# Step 1: Load the MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Replace with your MarianMT model
model = MarianMTModel.from_pretrained(model_name)

# Step 2: Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Step 3: Save the quantized model's state dictionary
torch.save(quantized_model.state_dict(), "quantized_marianmt.pth")
print("Quantized model saved as 'quantized_marianmt.pth'.")

# To load the model later on (or on a different device):
def load_quantized_model(model_name, state_dict_path):
    # Load the original model architecture
    model = MarianMTModel.from_pretrained(model_name)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(state_dict_path))
    
    # Apply dynamic quantization again to match the saved model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

# Step 4: Reload the quantized model
reloaded_quantized_model = load_quantized_model(model_name, "quantized_marianmt.pth")
print("Quantized model reloaded successfully.")

