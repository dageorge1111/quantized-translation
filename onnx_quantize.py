from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "marianmt_model.onnx",
    "marianmt_quantized.onnx",
    weight_type=QuantType.QUInt8
)
print("Quantized model saved as 'marianmt_quantized.onnx'.")

