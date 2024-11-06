import torch
import numpy as np
import ai_edge_torch
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).eval()

# Define a wrapper class to disable torch._dynamo around the forward pass
class DynamoDisabledModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# Wrap the model
wrapped_model = DynamoDisabledModelWrapper(model)

# Define sample input for tracing
sample_text = "This is a sample translation input."
sample_inputs = tokenizer(sample_text, return_tensors="pt")
input_ids = sample_inputs['input_ids']
attention_mask = sample_inputs['attention_mask']

# Run the conversion
edge_model = ai_edge_torch.convert(wrapped_model.eval(), (input_ids, attention_mask))

# Test inference with the converted model
edge_output = edge_model((input_ids, attention_mask))

# Check if outputs are within acceptable tolerance
torch_output = model(input_ids=input_ids, attention_mask=attention_mask)
if np.allclose(torch_output.logits.detach().numpy(), edge_output, atol=1e-5, rtol=1e-5):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something is wrong with Pytorch to TfLite conversion")

# Export the model as TFLite Flatbuffers file
edge_model.export("translation_model.tflite")
print("TFLite model exported successfully!")

