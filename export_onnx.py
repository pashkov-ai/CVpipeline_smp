import pytorch_lightning as pl
import torch
from cvpipeline_smp.lightning_module import SMPLightningModule
from cvpipeline_smp.lightning_module import SMPLightningModuleMultiClass
from omegaconf import OmegaConf

def export_pth(cfg_path: str, checkpoint_path: str) -> None:
    # Load Hydra config
    cfg = OmegaConf.load(cfg_path)

    # Load checkpoint with cfg parameter
    checkpoint = SMPLightningModuleMultiClass.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg
    )
    input_sample = torch.randn(16, 3, 256, 256)
    ret = checkpoint.to_onnx("onnx/model.onnx",
                             input_sample,
                             input_names=['image'],
                             output_names=['mask'],
                             export_params=True)
    print('Model exported successfully')


if __name__ == "__main__":
    cfg_path = "/home/user/Downloads/config.yaml"
    ckpt_path = "/home/user/Downloads/epoch=19-valid_loss=0.312415.ckpt"
    export_pth(cfg_path, ckpt_path)
