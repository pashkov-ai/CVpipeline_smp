import albumentations as A
from omegaconf import DictConfig


def get_transforms(mode: str, cfg: DictConfig) -> list:
    transforms = [
        A.OneOf(
            [
                A.RandomCrop(height=256, width=256, p=0.6),
                A.CropNonEmptyMaskIfExists(height=256, width=256, p=1),
            ],
            p=1.0,
        )
    ]

    if mode == "train":
        transforms.extend([
            # A.Resize(height=cfg.datamodule.image_size.height, width=cfg.datamodule.image_size.width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
            # A.RandomBrightnessContrast(p=0.2),
            ])
    if mode == "validation":
        transforms.extend([            # A.Resize(height=self.cfg.datamodule.image_size.height, width=self.cfg.datamodule.image_size.width),
            A.OneOf(
                [
                    A.RandomCrop(height=256, width=256, p=0.6),
                    A.CropNonEmptyMaskIfExists(height=256, width=256, p=1),
                ],
                p=1.0,
            )
        ])
    return transforms