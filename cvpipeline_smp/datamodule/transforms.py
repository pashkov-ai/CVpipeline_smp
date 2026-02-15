import albumentations as A
from omegaconf import DictConfig


def get_transforms(mode: str, cfg: DictConfig) -> list:
    transforms = [
    ]

    if mode == "train":
        transforms.extend([
            A.OneOf(
                [
                    A.RandomCrop(height=256, width=256, p=0.05),
                    A.CropNonEmptyMaskIfExists(height=256, width=256, p=1),
                ],
                p=1.0,
            ),
            # A.Resize(height=cfg.datamodule.image_size.height, width=cfg.datamodule.image_size.width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            ])

    if mode == "validation":
        transforms.extend([
            # A.Resize(height=self.cfg.datamodule.image_size.height, width=self.cfg.datamodule.image_size.width),
            A.RandomCrop(height=256, width=256, p=1.0)
        ])

    return transforms