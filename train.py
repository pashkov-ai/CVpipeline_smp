"""Training script for binary classification model using PyTorch Lightning."""
import torch
import hydra
import json
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cvpipeline_smp.datamodule.datamodule import AITEXFabricDataModule
from cvpipeline_smp.lightning_module import SMPLightningModule, SMPLightningModuleMultiClass

import os
import random
import numpy as np
import shutil
from pathlib import Path
import logging
from hydra.core.hydra_config import HydraConfig

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
    torch.use_deterministic_algorithms(True, warn_only=True)


def training_pipeline(cfg: DictConfig):
    """Train the binary classification model for fabric defect detection.

    Example:
        Run this script from the command line:
        $ python train.py
    """

    set_random_seed(cfg.general.random_seed)

    torch.set_float32_matmul_precision('high')

    # Setup logger
    logger = hydra.utils.instantiate(cfg.logger)

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    config_artifact_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_artifact_path)
    logger.experiment.log_artifact(
        run_id=logger.run_id,
        local_path=config_artifact_path
    )

    # log transforms
    logger.experiment.log_artifact(
        run_id=logger.run_id,
        local_path="cvpipeline_smp/datamodule/transforms.py"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        **cfg.callbacks.model_checkpoint
    )
    early_stopping_callback = EarlyStopping(**cfg.callbacks.early_stopping)

    other_callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks.other_callbacks]

    # todo: log augs to mlflow

    # Initialize trainer
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback] + other_callbacks,
        logger=logger,
        **cfg.trainer.params
    )
    trainer.logger.log_hyperparams(OmegaConf.to_object(cfg))
    logging.info("Training started with config:\n%s", json.dumps(OmegaConf.to_object(cfg), indent=2))


    # Initialize data module
    datamodule = AITEXFabricDataModule(cfg=cfg)

    # Initialize model
    lightning_model = None
    if cfg.labels.mode == "binary":
        lightning_model = SMPLightningModule(cfg=cfg)
    if cfg.labels.mode == 'multiclass':
        lightning_model = SMPLightningModuleMultiClass(cfg=cfg)

    # Train the model
    trainer.fit(lightning_model, datamodule=datamodule)

    # Test the model
    # trainer.test(lightning_model, datamodule=datamodule)


@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def hydra_run_train(cfg: DictConfig) -> None:
    training_pipeline(cfg)


if __name__ == "__main__":
    hydra_run_train()
