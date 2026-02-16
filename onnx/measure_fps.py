import time
import random
import numpy as np
import onnxruntime as ort
from pathlib import Path
import warnings
import logging

import platform
import psutil
import subprocess

import cv2

from load_dataset import load_dataset
from evaluate_model import predict_full_image_batched

# Suppress ONNX Runtime warnings about CUDA provider
logging.getLogger('onnxruntime').setLevel(logging.ERROR)


def benchmark_prediction(
    image_pathes: list[str] | list[Path],
    model_path: str | Path,
    device: str = 'cpu',
) -> dict[str, float | str]:
    """Benchmark the speed of predict_full_image_batched function.

    Runs the prediction function multiple times and calculates timing statistics
    to measure inference performance on CPU or GPU.

    Args:
        image_pathes: List of paths to input images (expected shape: 256×4096 each).
        model_path: Path to the ONNX model file.
        device: Device to run inference on, either 'cpu' or 'gpu' (default: 'cpu').

    Returns:
        Dictionary containing timing statistics:
            - 'device': Device used for inference ('cpu' or 'gpu').
            - 'providers': List of execution providers used by ONNX Runtime.
            - 'total_time': Total time for all iterations (seconds).
            - 'mean_time': Mean time per iteration (seconds).
            - 'min_time': Minimum time per iteration (seconds).
            - 'max_time': Maximum time per iteration (seconds).
            - 'std_time': Standard deviation of times (seconds).
            - 'fps': Frames per second (1 / mean_time).
            - 'num_iterations': Number of iterations performed.

    Raises:
        ValueError: If device is invalid or images cannot be loaded.

    Example:
        >>> image_paths = ["img1.png", "img2.png", "img3.png"]
        >>> # Benchmark on CPU
        >>> stats = benchmark_prediction(image_paths, "model.onnx", device='cpu')
        >>> print(f"Mean time: {stats['mean_time']:.4f}s")
        Mean time: 0.0234s
        >>> print(f"FPS: {stats['fps']:.2f}")
        FPS: 42.74
        >>> # Benchmark on GPU
        >>> stats_gpu = benchmark_prediction(image_paths, "model.onnx", device='gpu')
    """
    if device not in ['cpu', 'gpu']:
        raise ValueError(f"device must be 'cpu' or 'gpu', got '{device}'")

    # Configure execution providers based on device
    if device == 'gpu':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # Create session with specified providers
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Suppress ONNX Runtime warnings during session creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model_session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX session with {device}: {e}") from e

    # Get actual providers used (may fall back to CPU if GPU not available)
    actual_providers = model_session.get_providers()

    # Print device information
    print("=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Device requested: {device.upper()}")
    print(f"Execution providers: {actual_providers}")
    print(f"Number of images: {len(image_pathes)}")
    print(f"Model path: {model_path}")

    # Check if GPU was requested but not available
    if device == 'gpu' and 'CUDAExecutionProvider' not in actual_providers:
        print("\nWARNING: GPU (CUDA) requested but not available. Falling back to CPU.")

    print("=" * 60)

    # Store timing results
    times = []

    print(f"\nRunning benchmark with {len(image_pathes)} images on {device.upper()}...")

    # Run predictions and measure time
    for i, image_path in enumerate(image_pathes):
        start_time = time.perf_counter()

        _ = predict_full_image_batched(
            image_path=image_path,
            model_session=model_session,
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(image_pathes)} iterations...")

    # Convert to numpy array for statistics
    times_array = np.array(times)

    # Calculate statistics
    total_time = float(np.sum(times_array))
    mean_time = float(np.mean(times_array))
    min_time = float(np.min(times_array))
    max_time = float(np.max(times_array))
    std_time = float(np.std(times_array))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    print(f"Providers: {actual_providers}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Mean time: {mean_time:.4f}s")
    print(f"Min time: {min_time:.4f}s")
    print(f"Max time: {max_time:.4f}s")
    print(f"Std dev: {std_time:.4f}s")
    print(f"FPS: {fps:.2f}")
    print("=" * 60)

    return {
        'device': device,
        'providers': actual_providers,
        'total_time': total_time,
        'mean_time': mean_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'fps': fps,
        'num_iterations': len(image_pathes),
    }


if __name__ == "__main__":
    N_ITERATIONS = 100

    model_path = "model.onnx"

    dataset = load_dataset()

    pathes = [x for (_, x, _) in dataset['non_defect']]
    for k, v in dataset['defect'].items():
        pathes.extend(x for (_, x, _) in dataset['defect'][k])

    filtered_pathes = []
    for fp in pathes:
        img = cv2.imread("image.jpg")
        h, w = img.shape[:2]
        if w == 4096 and h == 256:
            filtered_pathes.append(fp)
        else:
            print(f"Skipping {fp} with shape {img.shape}")

    random.shuffle(filtered_pathes)

    assert len(filtered_pathes) > N_ITERATIONS

    # Benchmark on GPU
    out = subprocess.check_output(["nvidia-smi"], text=True)

    lines = out.splitlines()
    print('\n'.join(lines[:12]))
    stats_gpu = benchmark_prediction(filtered_pathes[:N_ITERATIONS], model_path, device='gpu')

    # Benchmark on CPU
    print("Processor:", platform.processor())
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Logical cores:", psutil.cpu_count(logical=True))
    print("Max frequency (MHz):", psutil.cpu_freq().max)
    stats_cpu = benchmark_prediction(filtered_pathes[:N_ITERATIONS], model_path, device='cpu')
