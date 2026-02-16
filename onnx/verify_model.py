import onnx
import onnxruntime as ort
import numpy as np


def verify_runtime_shape(model_path: str) -> None:
    # Load model
    sess = ort.InferenceSession("model.onnx")

    # Inspect input name & shape
    input_name = sess.get_inputs()[0].name
    print(input_name,sess.get_inputs()[0].shape)

    # Create dummy input (match model input shape!)
    x = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Run inference
    outputs = sess.run(None, {input_name: x})
    print('Outputs:', type(outputs), len(outputs), outputs[0].shape)


def verify_model(model_path: str) -> None:
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("Model verified successfully")


if __name__ == "__main__":
    model_path = "model.onnx"
    verify_model(model_path)
    verify_runtime_shape(model_path)
