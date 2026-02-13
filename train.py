"""Training script for binary classification model using PyTorch Lightning."""
from typing import Any

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cvpipeline_smp.config.training_config import TrainingConfig
from cvpipeline_smp.data.datamodule import AITEXFabricDataModule
from cvpipeline_smp.lightning_module import SMPLightningModule


def configure_optimizers(self) -> dict[str, Any]:
    """Configure optimizer and learning rate scheduler.

    Returns:
        Dictionary containing optimizer and scheduler configuration.
    """
    optimizer = Adam(self.parameters(), lr=self.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
        },
    }

def main() -> None:
    """Train the binary classification model for fabric defect detection.

    Example:
        Run this script from the command line:
        $ python train.py
    """
    # Load configuration
    config = TrainingConfig()

    # Initialize data module
    datamodule = AITEXFabricDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    # Initialize model
    lightning_model = SMPLightningModule()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="classification-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Setup logger
    logger = CSVLogger("logs", name="classification_training")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        # callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train the model
    trainer.fit(lightning_model, datamodule=datamodule)

    # Test the model
    trainer.test(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
