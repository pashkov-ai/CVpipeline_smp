"""Dataset loading utilities for AITEX Fabric Image Database."""

import os
from pathlib import Path
from typing import Any


def load_dataset(path: str = '../data/AITEX_Fabric_Image_Database') -> dict[str, Any]:
    """Load the AITEX fabric image dataset with defect and non-defect image paths.

    The AITEX dataset contains fabric images with various defects and defect-free samples.
    This function indexes all images, organizing them by defect type, and returns paths
    instead of loaded arrays for memory efficiency.

    Args:
        path: Path to the AITEX dataset root directory containing 'Defect_images',
            'Mask_images', and 'NODefect_images' subdirectories.

    Returns:
        Dictionary with the following structure:
        {
            'defect': {
                'defect_code': [(image_name, full_image_path, full_mask_path), ...],
                ...
            },
            'non_defect': [(image_name, full_image_path, None), ...]
        }
        where:
        - defect_code: String representing the defect type (e.g., '002', '006')
        - image_name: Original filename (e.g., '0001_002_00.png')
        - full_image_path: Absolute path to the image file

    Raises:
        FileNotFoundError: If the specified path or required subdirectories don't exist.

    Example:
        >>> dataset = load_dataset()
        >>> print(f"Defect types: {list(dataset['defect'].keys())}")
        Defect types: ['002', '006', '010', '016', ...]
        >>> print(f"Images with defect '002': {len(dataset['defect']['002'])}")
        Images with defect '002': 15
        >>> print(f"Non-defect images: {len(dataset['non_defect'])}")
        Non-defect images: 140
    """
    dataset_path = Path(path)

    # Validate dataset structure
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    defect_dir = dataset_path / "Defect_images"
    mask_dir = dataset_path / "Mask_images"
    no_defect_dir = dataset_path / "NODefect_images"

    for required_dir in [defect_dir, mask_dir, no_defect_dir]:
        if not required_dir.exists():
            raise FileNotFoundError(f"Required directory not found: {required_dir}")

    # Initialize result structure
    dataset: dict[str, Any] = {"defect": {}, "non_defect": []}

    # Load defect images and masks
    defect_files = sorted([f for f in os.listdir(defect_dir) if f.endswith(".png")])

    for defect_file in defect_files:
        if '0044_019_04' in defect_file or '0097_030_03' in defect_file:  # 2 masks
            continue
        if '0100_025_08' in defect_file:  # no mask
            continue

        # Parse filename: nnnn_ddd_ff.png
        parts = defect_file.replace(".png", "").split("_")
        if len(parts) != 3:
            continue

        defect_code = parts[1]

        # Get full path to image
        img_path = str((defect_dir / defect_file).resolve())

        mask_file = defect_file.replace(".png", "_mask.png")
        mask_path = str((mask_dir / mask_file).resolve())

        # Group by defect code
        if defect_code not in dataset["defect"]:
            dataset["defect"][defect_code] = []

        dataset["defect"][defect_code].append((defect_file, img_path, mask_path))

    # Load non-defect images
    # Non-defect images are nested in subdirectories
    for root, _, files in os.walk(no_defect_dir):
        for file in sorted(files):
            if file.endswith(".png"):
                img_path = str((Path(root) / file).resolve())
                dataset["non_defect"].append((file, img_path, None))

    for k, v in dataset['defect'].items():
        dataset['defect'][k] = sorted(dataset['defect'][k], key=lambda x: x[0])
    dataset['non_defect'] = sorted(dataset['non_defect'], key=lambda x: x[0])
    return dataset