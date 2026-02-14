import pytorch_lightning as pl
import torch
from cvpipeline_smp.lightning_module import SMPLightningModule

def export_pth(checkpoint_path: str) -> None:
    checkpoint = SMPLightningModule.load_from_checkpoint(checkpoint_path)
    input_sample = torch.randn(1, 3, 256, 4096)
    ret = checkpoint.to_onnx("onnx/model.onnx",
                             input_sample,
                             input_names=['image'],
                             output_names=['mask'],
                             export_params=True)
    print('Model exported successfully')


if __name__ == "__main__":
    ckpt_path = "/mnt/mlflow_artifacts/1/887e67b1e02040059fbb2f9b3b1603fc/artifacts/epoch=epoch=0-valid_loss=valid_loss=0.9993/epoch=epoch=0-valid_loss=valid_loss=0.9993.ckpt"
    export_pth(ckpt_path)
