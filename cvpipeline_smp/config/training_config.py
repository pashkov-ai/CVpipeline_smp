"""Training configuration for classification and segmentation models."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration for classification/segmentation models.

    Attributes:
        model_name: Model architecture name.
        encoder_name: Backbone encoder name (for timm: use underscore, e.g. efficientnet_b0).
        encoder_weights: Pretrained weights for encoder.
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output classes (2 for binary classification).
        batch_size: Batch size for training.
        lr: Learning rate.
        max_epochs: Maximum number of training epochs.
        num_workers: DataLoader workers.
        image_size: Input image size (height, width).

    Example:
        >>> config = TrainingConfig()
        >>> config.encoder_name
        'efficientnet_b0'
        >>> config.image_size
        (256, 256)
    """

    model_name: str = "BinaryClassificationModel"
    encoder_name: str = "efficientnet_b0"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 2
    batch_size: int = 4
    lr: float = 1e-4
    max_epochs: int = 1
    num_workers: int = 4
    image_size: tuple[int, int] = (256, 4096) # (h, w)
