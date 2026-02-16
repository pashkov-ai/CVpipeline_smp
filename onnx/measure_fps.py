import time
import random
import numpy as np
import onnxruntime as ort
from pathlib import Path


from load_dataset import load_dataset
from evaluate_model import predict_full_image_batched


def benchmark_prediction(
    image_pathes: list[str] | list[Path],
    model_session: ort.InferenceSession,
) -> dict[str, float]:
    """Benchmark the speed of predict_full_image_batched function.

    Runs the prediction function multiple times and calculates timing statistics
    to measure inference performance.

    Args:
        image_path: Path to the input image (expected shape: 256×4096).
        model_session: ONNX Runtime inference session with loaded model.
        num_iterations: Number of times to run the prediction (default: 100).

    Returns:
        Dictionary containing timing statistics:
            - 'total_time': Total time for all iterations (seconds).
            - 'mean_time': Mean time per iteration (seconds).
            - 'min_time': Minimum time per iteration (seconds).
            - 'max_time': Maximum time per iteration (seconds).
            - 'std_time': Standard deviation of times (seconds).
            - 'fps': Frames per second (1 / mean_time).
            - 'num_iterations': Number of iterations performed.

    Raises:
        ValueError: If num_iterations is less than 1 or image cannot be loaded.

    Example:
        >>> session = ort.InferenceSession("model.onnx")
        >>> stats = benchmark_prediction("test_image.png", session, num_iterations=100)
        >>> print(f"Mean time: {stats['mean_time']:.4f}s")
        Mean time: 0.0234s
        >>> print(f"FPS: {stats['fps']:.2f}")
        FPS: 42.74
    """

    # Store timing results
    times = []


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

    return {
        'total_time': total_time,
        'mean_time': mean_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'fps': fps,
        'num_iterations': len(image_pathes),
    }


if __name__ == "__main__":
    model_path = "model.onnx"
    session = ort.InferenceSession(model_path)

    dataset = load_dataset()

    pathes = [x for (_, x, _) in dataset['non_defect']]
    for k, v in dataset['defect'].items():
        pathes.extend(x for (_, x, _) in dataset['defect'][k])
    random.shuffle(pathes)
    benchmark_prediction(pathes[:100], session)