"""Training script for binary classification model using PyTorch Lightning."""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

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
import logging

logging.basicConfig(level=logging.INFO)


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


def training_pipeline(cfg: DictConfig):
    """Train the binary classification model for fabric defect detection.

    Example:
        Run this script from the command line:
        $ python train.py
    """

    set_random_seed(cfg.general.random_seed)

    # Load configuration
    config = TrainingConfig()
    torch.set_float32_matmul_precision('high')
    smp_model_config = {

    }

    # Initialize data module
    datamodule = AITEXFabricDataModule(cfg=cfg)

    # Initialize model
    lightning_model = SMPLightningModule(cfg=cfg)


    # Setup checkpoints callback
    checkpoint_dir = Path("training/")
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
    OmegaConf.save(cfg, "training/config.yaml")
    logger.experiment.log_artifact(
        run_id=logger.run_id,
        local_path="training/config.yaml"
    )

    # todo: log augs to mlflow

    # Initialize trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        **cfg.trainer
    )
    trainer.logger.log_hyperparams(OmegaConf.to_object(cfg))

    # Train the model
    trainer.fit(lightning_model, datamodule=datamodule)

    # Test the model
    trainer.test(lightning_model, datamodule=datamodule)


@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def hydra_run_train(cfg: DictConfig) -> None:
    training_pipeline(cfg)


if __name__ == "__main__":
    hydra_run_train()
