"""LightningDataModule for AITEX fabric defect detection."""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.encoders import get_preprocessing_params

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def split_by_proportions(
        data: list,
        proportions: tuple[int, ...],
) -> list[list]:
    """Split a list into multiple non-overlapping sublists by given proportions.

    Divides the input list into sequential parts based on the proportions provided.
    The proportions should sum to 100 (or any constant value).

    Args:
        data: The list to be split.
        proportions: Tuple of integers representing proportions (e.g., (75, 15, 10)).
            Values should sum to 100 for percentages, or any constant for ratios.

    Returns:
        List of sublists where each sublist contains the corresponding proportion
        of the original data in sequence.

    Raises:
        ValueError: If proportions sum to zero or if data is empty.
        TypeError: If proportions contain non-numeric values.

    Example:
        >>> data = list(range(100))
        >>> train, val, test = split_by_proportions(data, (75, 15, 10))
        >>> len(train), len(val), len(test)
        (75, 15, 10)
        >>> train == list(range(0, 75))
        True
        >>> val == list(range(75, 90))
        True
        >>> test == list(range(90, 100))
        True
    """
    if not data:
        raise ValueError("Data list cannot be empty")

    if not proportions or sum(proportions) == 0:
        raise ValueError("Proportions must be non-empty and sum to a non-zero value")

    total = sum(proportions)
    data_length = len(data)

    # Calculate split indices using numpy for precision
    split_indices = np.round(
        np.cumsum(proportions) / total * data_length
    ).astype(int)

    # Ensure the last index equals data length to avoid loss of elements
    split_indices = np.append(split_indices[:-1], data_length)

    # Split the data sequentially
    result = []
    start = 0
    for end in split_indices:
        result.append(data[start:end])
        start = end

    return result


class AITEXFabricDataset(Dataset):
    """AITEX Fabric Defect Detection Dataset for binary classification.

    Binary classification task: defect (1) vs no-defect (0).

    Args:
        image_paths: List of paths to fabric images.
        labels: List of binary labels (0 for no-defect, 1 for defect).
        transform: Albumentations transform pipeline.

    Example:
        >>> dataset = AITEXFabricDataset(image_paths, labels)
        >>> sample = dataset[0]
        >>> sample["image"].shape
        torch.Size([3, 256, 4096])
        >>> sample["label"]
        tensor(0)
    """

    def __init__(
        self,
        images: list[tuple[int, Path, Path | None]], # (label, image_path, mask_path)
        n_labels: int,
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize the AITEX fabric dataset."""
        self.images = images
        self.n_labels = n_labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary containing:
                - "image": RGB image tensor of shape (3, H, W)
                - "label": Binary label tensor (0 or 1)
        """
        label, image_path, mask_path = self.images[idx]

        # label = torch.tensor(label, dtype=torch.long)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path and mask_path:
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
        else:
            # Create empty mask if mask file doesn't exist
            raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        mask = np.zeros_like(raw_mask)
        mask[raw_mask != 0] = label
        # print('getitem: ', image.shape, mask.shape)
        # Apply transforms
        # print('Before: ', image.shape, mask.shape)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0) # for binary
        # print('After: ', image.shape, mask.shape)
        return {"label": label, "image": image, "mask": mask}


class AITEXFabricDataModule(pl.LightningDataModule):
    """LightningDataModule for AITEX fabric defect detection.

    Binary classification: defect vs no-defect.
    Data split: 75% train, 15% validation, 10% test.

    Args:
        data_dir: Path to the AITEX_Fabric_Image_Database directory.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for dataloaders.
        image_size: Tuple of (height, width) for resizing images.
        seed: Random seed for train/val/test split.

    Example:
        >>> dm = AITEXFabricDataModule(data_dir="data/AITEX_Fabric_Image_Database")
        >>> dm.setup("fit")
        >>> batch = next(iter(dm.train_dataloader()))
        >>> batch["image"].shape
        torch.Size([batch_size, 3, 256, 256])
    """

    def __init__(
        self,
        data_dir: str | Path = "data/AITEX_Fabric_Image_Database",
        labels_mapping: dict[str, int] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: tuple[int, int] = (256, 256),
        model_config: dict = None,
        seed: int = 42,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.data_dir = data_dir
        self.labels_mapping = labels_mapping
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed

        self.train_dataset: AITEXFabricDataset | None = None
        self.val_dataset: AITEXFabricDataset | None = None
        self.test_dataset: AITEXFabricDataset | None = None

        # preprocessing parameteres for image
        params = get_preprocessing_params(encoder_name=model_config['encoder_name'], pretrained=model_config['encoder_weights'])
        # Define transforms
        # todo move transform defentions outside
        self.train_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=params['mean'], std=params['std']),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=params['mean'], std=params['std']),
            ToTensorV2(),
        ])

        self._prepare_datasets()

    def _load_dataset(self) -> dict[str, Any]:
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
        dataset_path = Path(self.data_dir)

        # Validate dataset structure
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_dir}")

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
            if '0044_019_04' in defect_file or '0097_030_03' in defect_file :# 2 masks
                continue
            if '0100_025_08' in defect_file: # no mask
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

    def _prepare_datasets(self):
        """Load all image paths and labels from the dataset.

        Returns:
            Tuple of (image_paths, labels) where labels are 0 (no-defect) or 1 (defect).
        """


        # todo infer / verify  mode (binary, multiclass, multilabel) here
        # convert raw dataset format to labeled dataset format
        dataset = self._load_dataset()
        labeled_dataset = {}
        # todo triyng to reduce amount of background
        # labeled_dataset = {0: []}
        # for (_, image_path, mask_path) in dataset['non_defect']:
        #     labeled_dataset[0].append((0, image_path, mask_path))

        if self.labels_mapping is None: # todo a hack for code belowto work
            labeled_dataset[1] = []

        for defect_code in sorted(dataset['defect'].keys()):
            if self.labels_mapping is None: # everything is one class 'defect'
                for (_, image_path, mask_path) in dataset['defect'][defect_code]:
                    labeled_dataset[1].append((1, image_path, mask_path))
            else:
                if defect_code in self.labels_mapping:
                    label = self.labels_mapping[defect_code]
                    labeled_dataset[label] = []
                    for (_, image_path, mask_path) in dataset['defect'][defect_code]:
                        labeled_dataset[label].append((label, image_path, mask_path))

        n_labels = len(labeled_dataset.keys())

        dataset_splits = [[], [], []]
        for label, label_list in labeled_dataset.items():
            splits = split_by_proportions(label_list, (75, 15, 10))
            dataset_splits[0].extend(splits[0])
            dataset_splits[1].extend(splits[1])
            dataset_splits[2].extend(splits[2])

        print(n_labels, len(labeled_dataset[1]), len(dataset_splits[0]), len(dataset_splits[1]), len(dataset_splits[2]))
        self.train_dataset = AITEXFabricDataset(
            images=dataset_splits[0],
            n_labels=n_labels,
            transform=self.train_transform,
        )
        self.val_dataset = AITEXFabricDataset(
            images=dataset_splits[1],
            n_labels=n_labels,
            transform=self.val_transform,
        )
        self.test_dataset = AITEXFabricDataset(
            images=dataset_splits[2],
            n_labels=n_labels,
            transform=self.val_transform,
        )


    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader for training data with shuffle=True.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader for validation data with shuffle=False.
        """

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            DataLoader for test data with shuffle=False.
        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
