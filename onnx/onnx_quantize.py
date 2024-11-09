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

# Get the decoder start token ID
decoder_start_token_id = config.decoder_start_token_id

# Prepare a small set of calibration data
calibration_inputs = []
sample_texts = ["Hello world!", "How are you?", "Good morning!"]  # Add more samples if possible
for text in sample_texts:
    inputs = tokenizer([text], return_tensors="np")
    batch_size = inputs['input_ids'].shape[0]
    # Initialize decoder_input_ids with decoder_start_token_id
    decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
    calibration_inputs.append({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'decoder_input_ids': decoder_input_ids
    })

# Quantize the model without 'optimize_model' parameter
quantize_static(
    model_input="marianmt_model.onnx",
    model_output="marianmt_quantized_static.onnx",
    calibration_data_reader=MyCalibrationDataReader(calibration_inputs),
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print("Quantized model saved as 'marianmt_quantized_static.onnx'.")

