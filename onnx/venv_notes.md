python3 -m venv onnx/CVpipeline-onnx
source onnx/CVpipeline-onnx/bin/activate
pip install onnx==1.20.0 onnxruntime onnxruntime-gpu opencv-python

pip freeze > onnx/requirements.txt
pip install -r onnx/requirements.txt