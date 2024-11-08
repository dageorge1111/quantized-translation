from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from transformers import MarianTokenizer, MarianConfig
import numpy as np

# Define the tokenizer and config
model_name = 'Helsinki-NLP/opus-mt-en-de'  # Replace with your desired model
tokenizer = MarianTokenizer.from_pretrained(model_name)
config = MarianConfig.from_pretrained(model_name)

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_inputs):
        self.data = calibration_inputs
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

# Prepare a small set of calibration data
calibration_inputs = []
sample_texts = ["Hello world!", "How are you?", "Good morning!"]  # Add more samples if possible
for text in sample_texts:
    inputs = tokenizer([text], return_tensors="np")
    calibration_inputs.append({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    })

# Quantize the model
quantize_static(
    model_input="marianmt_model.onnx",
    model_output="marianmt_quantized_static.onnx",
    calibration_data_reader=MyCalibrationDataReader(calibration_inputs),
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    optimize_model=False  # Set to True if you want to optimize the model during quantization
)

print("Quantized model saved as 'marianmt_quantized_static.onnx'.")

