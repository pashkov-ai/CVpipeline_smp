
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig


class SMPLightningModule(pl.LightningModule):
    """PyTorch Lightning module for binary segmentation.

    Uses UNet++ architecture with EfficientNet-B0 encoder from
    segmentation_models_pytorch library.

    Args:
        encoder_name: Name of the encoder backbone.
        encoder_weights: Pretrained weights for the encoder.
        in_channels: Number of input channels.
        classes: Number of output classes.
        lr: Learning rate for optimizer.

    Example:
        >>> model = SegmentationModel(lr=1e-4)
        >>> x = torch.rand(2, 3, 256, 256)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 1, 256, 256])
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the segmentation model."""
        super().__init__()
        self.cfg = cfg
        self.model = smp.create_model(
            in_channels=3,
            classes=cfg.labels.classes,
            # activation="sigmoid",
            **cfg.model
        )

        # for image segmentation dice loss could be the best first choice
        if cfg.labels.mode == 'binary':
            self.loss_fn = hydra_instantiate(cfg.loss, mode = smp.losses.BINARY_MODE)
        elif cfg.labels.mode == 'multiclass':
            raise ValueError(f"Unsupported mode: {cfg.labels.mode}")
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        elif cfg.labels.mode == 'multilabel':
            raise ValueError(f"Unsupported mode: {cfg.labels.mode}")
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        else:
            raise ValueError(f"Unsupported mode: {cfg.labels.mode}")

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, classes, H, W).
        """
        return self.model(x)

    def shared_step(self, batch, stage):
        image = batch['image']
        mask = batch['mask']

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4 # todo 4 for binary?

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch

        # print(image.shape, mask.shape, logits_mask.shape, prob_mask.shape, pred_mask.shape)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        if stage == "train":
            ret = {
                f"loss": loss,
                f"tp": tp,
                f"fp": fp,
                f"fn": fn,
                f"tn": tn,
            }
        else:
            ret = {
                f"{stage}_loss": loss,
                f"tp": tp,
                f"fp": fp,
                f"fn": fn,
                f"tn": tn,
            }
        return ret

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log('valid_loss', valid_loss_info['valid_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return



    def configure_optimizers(self):        # todo hydra config
        optimizer = hydra_instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = hydra_instantiate(self.cfg.scheduler, optimizer=optimizer)
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.cfg.general.scheduler_interval,
                "frequency": self.cfg.general.scheduler_frequency,
                "monitor": self.cfg.general.scheduler_monitor,
            },
        }
        return ret