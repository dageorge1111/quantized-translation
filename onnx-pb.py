from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model = onnx.load("marianmt_model.onnx")
tf_rep = prepare(onnx_model)
print("did stuff starting")
# Export to TensorFlow SavedModel format
tf_rep.export_graph("saved_model")
