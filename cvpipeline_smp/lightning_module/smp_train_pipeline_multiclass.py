
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import matplotlib.figure


class SMPLightningModuleMultiClass(pl.LightningModule):
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
        if cfg.labels.mode == 'multiclass':
            self.loss_fn = hydra_instantiate(cfg.loss, mode=smp.losses.MULTICLASS_MODE)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # visualization samples storage
        self.train_vis_samples = []
        self.valid_vis_samples = []
        self.max_vis_samples = 4  # Number of samples to visualize per epoch

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
        mask = batch['mask'].long()

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

        assert mask.ndim == 3

        logits_mask = self.forward(image)
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)
        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # print(image.shape, mask.shape, logits_mask.shape, prob_mask.shape, pred_mask.shape)
        torch.use_deterministic_algorithms(False) # todo a hack for  _histc_cuda does not have a deterministic implementation
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="multiclass", num_classes=self.cfg.labels.classes
        )
        torch.use_deterministic_algorithms(True)
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

    def visualize_predictions(
        self,
        images: torch.Tensor,
        masks_true: torch.Tensor,
        masks_pred: torch.Tensor,
        max_samples: int = 4,
    ):
        """Create visualization of predictions for multiclass segmentation.

        Args:
            images: Batch of input images of shape (B, C, H, W).
            masks_true: Batch of ground truth masks of shape (B, H, W) with class indices.
            masks_pred: Batch of predicted masks of shape (B, H, W) with class indices.
            max_samples: Maximum number of samples to visualize.

        Returns:
            Matplotlib figure containing the visualizations.

        Example:
            >>> fig = self.visualize_predictions(images, masks_true, masks_pred, 4)
            >>> self.logger.experiment.log_figure(fig, 'valid/predictions.png')
        """
        num_samples = min(max_samples, images.shape[0])
        num_classes = self.cfg.labels.classes

        # Move tensors to CPU and convert to numpy
        images = images[:num_samples].detach().cpu()
        masks_true = masks_true[:num_samples].detach().cpu()
        masks_pred = masks_pred[:num_samples].detach().cpu()

        # Create figure with subplots: each row shows [image, ground truth, prediction]
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # Use a categorical colormap for better class distinction
        cmap = plt.cm.get_cmap('tab10', num_classes)

        for idx in range(num_samples):
            # Get image (C, H, W) and denormalize if needed
            img = images[idx]

            # Convert from (C, H, W) to (H, W, C) for plotting
            if img.shape[0] == 3:  # RGB image
                img_np = img.permute(1, 2, 0).numpy()
                # Clip to [0, 1] range for visualization
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            else:
                img_np = img.squeeze().numpy()

            # Get masks (H, W) - class indices
            mask_true_np = masks_true[idx].numpy()
            mask_pred_np = masks_pred[idx].numpy()

            # Plot original image
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f"Input Image {idx + 1}", fontsize=12)
            axes[idx, 0].axis("off")

            # Plot ground truth mask with categorical colormap
            im1 = axes[idx, 1].imshow(mask_true_np, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation='nearest')
            axes[idx, 1].set_title("Ground Truth", fontsize=12)
            axes[idx, 1].axis("off")

            # Plot predicted mask with categorical colormap
            im2 = axes[idx, 2].imshow(mask_pred_np, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation='nearest')
            axes[idx, 2].set_title("Prediction", fontsize=12)
            axes[idx, 2].axis("off")

            # Add colorbar for the last row
            if idx == num_samples - 1:
                # Create colorbar with discrete class labels
                cbar = plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
                cbar.set_label('Class', rotation=270, labelpad=15)
                cbar.set_ticks(range(num_classes))
                cbar.set_ticklabels([f'Class {i}' for i in range(num_classes)])

        plt.tight_layout()
        return fig

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

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        self.log('train_loss', train_loss_info['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Store samples for visualization (only first batch)
        if batch_idx == 0 and len(self.train_vis_samples) == 0:
            image = batch['image']
            mask = batch['mask']
            logits_mask = self.forward(image)
            # For multiclass: softmax to get probabilities, then argmax to get class indices
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)

            self.train_vis_samples.append({
                'images': image.detach(),
                'masks_true': mask.detach(),
                'masks_pred': pred_mask.detach(),
            })

        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")

        # Log visualizations if samples were collected
        if len(self.train_vis_samples) > 0 and self.logger is not None:
            try:
                sample = self.train_vis_samples[0]
                fig = self.visualize_predictions(
                    images=sample['images'],
                    masks_true=sample['masks_true'],
                    masks_pred=sample['masks_pred'],
                    max_samples=self.max_vis_samples,
                )

                # Log figure to MLFlow
                if fig is not None:
                    self.logger.experiment.log_figure(
                        run_id=self.logger.run_id,
                        figure=fig,
                        artifact_file=f"train/predictions_epoch_{self.current_epoch}.png"
                    )
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: Failed to log train visualizations: {e}")
            finally:
                # Clear samples for next epoch
                self.train_vis_samples.clear()

        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log('valid_loss', valid_loss_info['valid_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Store samples for visualization (only first batch)
        if batch_idx == 0 and len(self.valid_vis_samples) == 0:
            image = batch['image']
            mask = batch['mask']
            logits_mask = self.forward(image)
            # For multiclass: softmax to get probabilities, then argmax to get class indices
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)

            self.valid_vis_samples.append({
                'images': image.detach(),
                'masks_true': mask.detach(),
                'masks_pred': pred_mask.detach(),
            })

        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")

        # Log visualizations if samples were collected
        if len(self.valid_vis_samples) > 0 and self.logger is not None:
            try:
                sample = self.valid_vis_samples[0]
                fig = self.visualize_predictions(
                    images=sample['images'],
                    masks_true=sample['masks_true'],
                    masks_pred=sample['masks_pred'],
                    max_samples=self.max_vis_samples,
                )

                # Log figure to MLFlow
                if fig is not None:
                    self.logger.experiment.log_figure(
                        run_id=self.logger.run_id,
                        figure=fig,
                        artifact_file=f"valid/predictions_epoch_{self.current_epoch}.png"
                    )
                    plt.close(fig)
            except Exception as e:
                print(f"Warning: Failed to log validation visualizations: {e}")
            finally:
                # Clear samples for next epoch
                self.valid_vis_samples.clear()

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