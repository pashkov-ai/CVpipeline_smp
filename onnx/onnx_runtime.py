import onnxruntime as ort
import numpy as np
import cv2

def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model using ONNX Runtime and return InferenceSession object."""
    return ort.InferenceSession(model_path)

def load_image(image_path: str) -> np.ndarray:
    """Load image using OpenCV and return as numpy array."""
    # todo: add image preprocessing
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == "__main__":
    model_path = "model.onnx"
    model_session = load_model(model_path)
    image = load_image("test.jpg")
    model_session.run(None, {'image': image})