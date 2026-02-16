# ONNX Model Inference

## Setup
0. Download the ONNX model from the [here](https://drive.google.com/file/d/1e_ba2fC5sDoV03nS5HHaA7JBGotJR59w/view?usp=drive_link)
1. Place `model.onnx` in this directory
2. Install dependencies: `pip install -r requirements.txt`
3. See `venv_notes` for additional environment setup reference

## Scripts

- **`verify_model.py`**: Verify ONNX model structure and validity
- **`evaluate_model.py`**: Run inference on an image and generate a prediction mask
- **`measure_fps.py`**: Benchmark model FPS on CPU/GPU (if available)

## Performance

See `benchmark.txt` for reference performance measurements
