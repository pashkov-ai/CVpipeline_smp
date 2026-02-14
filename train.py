"""Training script for binary classification model using PyTorch Lightning."""
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from cvpipeline_smp.config.training_config import TrainingConfig
from cvpipeline_smp.data.datamodule import AITEXFabricDataModule
from cvpipeline_smp.lightning_module import SMPLightningModule

import os
import random
import numpy as np
import shutil
from pathlib import Path


def set_random_seed(seed: int = 8888) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def training_pipeline():
    """Train the binary classification model for fabric defect detection.

    Example:
        Run this script from the command line:
        $ python train.py
    """

    set_random_seed()

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


    # Setup checkpoints callback
    checkpoint_dir = Path("training/checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="training/checkpoints",
        filename="epoch={epoch}-valid_loss={valid_loss:.4f}",
        monitor="valid_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Setup logger
    logger = MLFlowLogger(
        experiment_name="localtest",
        tracking_uri="http://127.0.0.1:5000/",
        log_model=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,

        deterministic=True,
        benchmark=False
    )
    trainer.logger.log_hyperparams(smp_model_config)

    # Train the model
    trainer.fit(lightning_model, datamodule=datamodule)

    # Test the model
    trainer.test(lightning_model, datamodule=datamodule)


def main() -> None:
    training_pipeline()


if __name__ == "__main__":
    main()
