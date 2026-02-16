"""ONNX model testing utilities for multiclass segmentation.

This module provides functions to test ONNX models on full-size images
by splitting them into tiles, running batch inference, and comparing
predictions against ground truth masks.
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from load_dataset import load_dataset



def predict_full_image_batched(
    image_path: str | Path,
    model_session: ort.InferenceSession,
    tile_size: int = 256,
    input_width: int = 4096,
) -> np.ndarray:
    """Split image into tiles, run batch inference, and stitch results.

    Takes a 256×4096 image, splits it into 256×256 tiles, runs multiclass
    segmentation inference through an ONNX model in batch mode, and stitches
    the predictions back into a single mask.

    Args:
        image_path: Path to the input image (expected shape: 256×4096).
        model_session: ONNX Runtime inference session with loaded model.
        encoder_name: Name of the encoder used in the model (for preprocessing).
        encoder_weights: Pretrained weights identifier (for preprocessing params).
        tile_size: Size of each square tile (default: 256).
        input_width: Expected width of the input image (default: 4096).

    Returns:
        Numpy array of shape (256, 4096) containing predicted class indices
        for multiclass segmentation (0=background, 1=class_002, 2=class_006, 3=class_010).

    Raises:
        ValueError: If image cannot be loaded or has incorrect dimensions.
        RuntimeError: If ONNX inference fails.

    Example:
        >>> session = ort.InferenceSession("model.onnx")
        >>> mask = predict_full_image_batched("test_image.png", session)
        >>> mask.shape
        (256, 4096)
        >>> np.unique(mask)
        array([0, 1, 2, 3])
    """
    # Load and validate image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channels = image.shape
    if height != tile_size:
        raise ValueError(
            f"Image height must be {tile_size}, got {height}"
        )
    if width != input_width:
        raise ValueError(
            f"Image width must be {input_width}, got {width}"
        )

    # Calculate number of tiles
    num_tiles = width // tile_size

    # Split image into tiles
    tiles = []
    for i in range(num_tiles):
        x_start = i * tile_size
        x_end = x_start + tile_size
        tile = image[:, x_start:x_end, :]
        tiles.append(tile)

    # Get preprocessing parameters for the encoder
    params = {
        'mean': np.array([0.485, 0.456, 0.406]),
        'std': np.array([0.229, 0.224, 0.225])
    }
    # Preprocess tiles: normalize and convert to tensor format
    preprocessed_tiles = []
    for tile in tiles:
        # Normalize using encoder-specific mean and std
        tile_normalized = tile.astype(np.float32) / 255.0
        tile_normalized = (tile_normalized - params['mean']) / params['std']

        # Convert from HWC to CHW format
        tile_chw = np.transpose(tile_normalized, (2, 0, 1))
        preprocessed_tiles.append(tile_chw)

    # Stack tiles into batch (batch_size, channels, height, width)
    batch = np.stack(preprocessed_tiles, axis=0).astype(np.float32)

    # Get input name for the model
    input_name = model_session.get_inputs()[0].name

    # Run batch inference
    outputs = model_session.run(None, {input_name: batch})
    logits = outputs[0]  # Shape: (num_tiles, num_classes, H, W)
    # Run inference on each tile individually and collect outputs
    # tile_outputs = []
    # for i, tile in enumerate(preprocessed_tiles):
    #     # Add batch dimension: (1, channels, height, width)
    #     tile_input = np.expand_dims(tile, axis=0).astype(np.float32)
    #
    #     # Run inference on single tile
    #     output = model_session.run(None, {input_name: tile_input})
    #     logits = output[0]  # Shape: (1, num_classes, H, W)
    #
    #     # Remove batch dimension: (num_classes, H, W)
    #     tile_outputs.append(logits[0])
    # Stack all tile outputs together
    # logits = np.stack(tile_outputs, axis=0)


    # Apply softmax to get probabilities
    # For numerical stability, subtract max before exp
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Get predicted class indices (argmax over class dimension)
    pred_masks = np.argmax(probs, axis=1)  # Shape: (num_tiles, H, W)

    # Stitch tiles back together horizontally
    full_mask = np.concatenate(pred_masks, axis=1)  # Shape: (H, W_total)

    return full_mask


def test_prediction_against_mask(
    test_case: tuple[str, str | Path, str | Path | None],
    model_session: ort.InferenceSession,
    raw_mask_px_th: int = 0,
) -> dict[str, float | bool]:
    """Test model prediction against ground truth mask.

    Takes a test case tuple, uses the model to predict a mask for the image,
    and compares it against the ground truth mask. If the mask path is None,
    assumes all pixels should be background (class 0).

    Args:
        test_case: Tuple of (error_code, image_path, mask_path).
            - error_code: Label/class ID for defect pixels (1, 2, or 3).
            - image_path: Path to the input image.
            - mask_path: Path to ground truth mask, or None for all background.
        model_session: ONNX Runtime inference session with loaded model.
        encoder_name: Name of the encoder used in the model (for preprocessing).
        encoder_weights: Pretrained weights identifier (for preprocessing params).
        raw_mask_px_th: Threshold for converting raw mask to binary (default: 0).

    Returns:
        Dictionary containing test metrics:
            - 'pixel_accuracy': Overall pixel classification accuracy (0-1).
            - 'iou_per_class': List of IoU scores for each class.
            - 'mean_iou': Mean IoU across all classes.
            - 'passed': Boolean indicating if test passed (accuracy > 0.8).

    Raises:
        ValueError: If image or mask cannot be loaded.
        RuntimeError: If prediction fails.

    Example:
        >>> session = ort.InferenceSession("model.onnx")
        >>> test_case = (1, "image.png", "mask.png")
        >>> results = test_prediction_against_mask(test_case, session)
        >>> results['pixel_accuracy']
        0.95
        >>> results['passed']
        True
    """
    error_code, image_path, mask_path = test_case

    mapping = {
      'background': 0,
      '002': 1,
      '006': 2,
      '010': 3
    }

    # Get prediction using the batched inference function
    pred_mask = predict_full_image_batched(
        image_path=image_path,
        model_session=model_session,
    )

    # Load or create ground truth mask
    if mask_path is not None:
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Convert raw mask to class labels
        # Pixels above threshold get the error_code, others stay 0 (background)
        gt_mask = np.zeros_like(raw_mask, dtype=np.int64)
        gt_mask[raw_mask > raw_mask_px_th] = mapping[error_code]
    else:
        # No mask provided: all pixels should be background (class 0)
        gt_mask = np.zeros(pred_mask.shape, dtype=np.int64)

    # Ensure shapes match
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred_mask.shape} vs "
            f"ground truth {gt_mask.shape}"
        )

    # Calculate pixel accuracy
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = pred_mask.size
    pixel_accuracy = correct_pixels / total_pixels

    # Calculate IoU per class
    # Infer number of classes from the data
    num_classes = max(np.max(pred_mask), np.max(gt_mask)) + 1
    iou_per_class = []

    for class_id in range(num_classes):
        # True positives: pixels correctly predicted as this class
        tp = np.sum((pred_mask == class_id) & (gt_mask == class_id))

        # False positives: pixels incorrectly predicted as this class
        fp = np.sum((pred_mask == class_id) & (gt_mask != class_id))

        # False negatives: pixels of this class predicted as something else
        fn = np.sum((pred_mask != class_id) & (gt_mask == class_id))

        # IoU = TP / (TP + FP + FN)
        union = tp + fp + fn
        if union > 0:
            iou = tp / union
        else:
            # If class doesn't exist in both pred and gt, count as perfect
            iou = 1.0

        iou_per_class.append(float(iou))

    mean_iou = float(np.mean(iou_per_class))

    # Determine if test passed (using 0.8 threshold for pixel accuracy)
    passed = pixel_accuracy > 0.8

    return {
        'pixel_accuracy': float(pixel_accuracy),
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou,
        'passed': passed,
    }


if __name__ == "__main__":
    """Example usage of the test functions."""
    # Load the ONNX model
    model_path = "model.onnx"
    session = ort.InferenceSession(model_path)

    dataset = load_dataset()

    idx = -1
    (_, img_path, mask_path)  = dataset['defect']['006'][idx]
    pmask = predict_full_image_batched(
        img_path,
        session
    )




    # results = test_prediction_against_mask(
    #     test_case=('002', img_path, mask_path),
    #     model_session=session,
    #     raw_mask_px_th=0,
    # )
    # print(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")
    # print(f"Mean IoU: {results['mean_iou']:.4f}")
    # print(f"IoU per class: {results['iou_per_class']}")
    # print(f"Test passed: {results['passed']}")