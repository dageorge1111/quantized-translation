import torch
import numpy as np
import onnx
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

# Create dummy input for ONNX export
dummy_input = tokenizer("This is a test sentence.", return_tensors="pt").input_ids

# Export the model to ONNX in float32 precision
onnx_model_path = "../models/marianmt_model_float32.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={"input_ids": {0: "batch_size"}}
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
    # Greetings and basic phrases
    "Hello, how are you?",
    "Good morning!",
    "Thank you very much.",
    "What time is it?",
    "Have a great day!",

    # Informal conversation
    "What are you up to today?",
    "I’m really looking forward to the weekend.",
    "Can we catch up later?",
    "I’ll see you soon!",
    "Let’s grab a coffee sometime.",

    # Travel-related phrases
    "Where is the nearest hotel?",
    "How do I get to the train station?",
    "I would like to book a flight.",
    "What’s the best way to reach downtown?",
    "Is there a restaurant nearby?",

    # Directions and locations
    "Turn left at the next intersection.",
    "Go straight for two blocks and then turn right.",
    "Can you show me on the map?",
    "Is it within walking distance?",
    "It’s located just past the bridge.",

    # Shopping phrases
    "How much does this cost?",
    "Do you have this in a different size?",
    "I’m just browsing, thank you.",
    "Can I get a receipt, please?",
    "What are your store hours?",

    # Technical queries
    "What is the latest version of this software?",
    "How can I reset my password?",
    "Where can I find documentation for this API?",
    "Is there a way to optimize this code?",
    "Can you help me troubleshoot this issue?",

    # Work-related language
    "Can we schedule a meeting for tomorrow?",
    "Please review the document and provide feedback.",
    "What are the project requirements?",
    "I need help with this task.",
    "Let’s work on a solution together.",

    # Educational and factual statements
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "The capital of France is Paris.",
    "An ecosystem consists of all living things in a particular area.",
    "Photosynthesis is the process by which plants make their food.",

    # Travel and tourism descriptions
    "The city has beautiful historic architecture.",
    "The mountains offer a stunning view of the valley.",
    "Many tourists visit this area for its beaches.",
    "This place is known for its unique cultural heritage.",
    "The local cuisine is famous worldwide.",

    # Weather and seasons
    "Today’s forecast is partly cloudy with a chance of rain.",
    "It gets quite cold here during the winter.",
    "Summers are usually very hot and humid.",
    "The temperature is expected to drop overnight.",
    "It rains frequently in this region.",

    # Miscellaneous statements
    "I enjoy reading books and watching movies.",
    "This recipe is easy to follow and tastes delicious.",
    "The concert was a lot of fun!",
    "It’s important to stay hydrated, especially in the summer.",
    "Technology is advancing at a rapid pace.",

    # Questions
    "What is the capital of Germany?",
    "How does this system work?",
    "Why is the sky blue?",
    "Can you explain this to me?",
    "What’s the best way to learn a new language?",

    # Opinions and statements
    "In my opinion, this is the best solution.",
    "I think it’s a great idea.",
    "That movie was very entertaining.",
    "I really enjoy this hobby.",
    "Learning a new language is challenging but rewarding."
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
quantized_model_path = "../models/marianmt_quantized_static.onnx"
quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=MyCalibrationDataReader(calibration_inputs),
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print(f"Quantized model saved as '{quantized_model_path}'.")

