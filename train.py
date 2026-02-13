"""Training script for binary classification model using PyTorch Lightning."""
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cvpipeline_smp.config.training_config import TrainingConfig
from cvpipeline_smp.data.datamodule import AITEXFabricDataModule
from cvpipeline_smp.lightning_module import SMPLightningModule


def training_pipeline():
    """Train the binary classification model for fabric defect detection.

    Example:
        Run this script from the command line:
        $ python train.py
    """
    # Load configuration
    config = TrainingConfig()
    torch.set_float32_matmul_precision('high')
    smp_model_config = {
        'arch': 'UnetPlusPlus',
        'encoder_name': 'efficientnet-b0',
        'encoder_weights': "imagenet",
        'in_channels': 3,
        'classes': 1
    }

    # Initialize data module
    datamodule = AITEXFabricDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        model_config=smp_model_config,
    )

    # Initialize model
    lightning_model = SMPLightningModule(smp_model_config, mode='binary')

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

def main() -> None:
    training_pipeline()


if __name__ == "__main__":
    main()
