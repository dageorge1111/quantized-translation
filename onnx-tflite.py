from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model = onnx.load("marianmt_model.onnx")
tf_rep = prepare(onnx_model)

# Export to TensorFlow SavedModel format
tf_rep.export_graph("saved_model")

import tensorflow as tf

# Load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

# Apply optimizations for size reduction
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # You can also use tf.int8 for full integer quantization

# Convert the model
tflite_model = converter.convert()

# Save the TF Lite model to a file
with open("marianmt_model.tflite", "wb") as f:
    f.write(tflite_model)

