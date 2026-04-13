import os
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import torch
from datamodule import CTDataModule
from models import LEARN_pl


def load_callbacks(n_iter, n_view, noise):
    """
    Create callback list for training.

    Notes:
    - Logic is kept identical to the original implementation.
    - Output path naming, monitor metric, and checkpoint behavior are unchanged.
    """
    callbacks = []

    output_path = (
        "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/"
        "LEARN_LongNet/saved_results_noise_2_with_LongNet/"
        "results_LEARN_"
        + str(n_iter)
        + "_iters_bs_1_view_"
        + str(n_view)
        + "_noise_"
        + str(noise)
        + "_transform/"
    )
    os.makedirs(output_path, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_psnr",  # val_ssim, val_psnr, val_rmse
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="max",
    )
    callbacks.append(early_stopping)

    checkpoint = ModelCheckpoint(
        monitor="val_psnr",
        dirpath=output_path,
        filename="{epoch:02d}-{val_psnr:.4f}",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
        save_weights_only=True,
    )
    callbacks.append(checkpoint)

    return callbacks


def main():
    """Main training entrypoint. Training logic is unchanged."""
    torch.manual_seed(42)

    num_view = 18
    input_size = 256
    num_detectors = 512
    poission_level = 5e5

    setting = (
        "numview_"
        + str(num_view)
        + "_inputsize_256_noise_0"
    )
    path_dir = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split/"

    n_iterations = 14
    batch_size = 1

    n_iter, n_view, noise = n_iterations, num_view, str(poission_level)
    print("n_iter, n_view, noise", n_iter, n_view, noise)

    seed_everything(42, workers=True)
    tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")

    checkpoint_path = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_18_noise_500000.0_transform/epoch=10-val_psnr=6.3106.ckpt"
    resume = True

    # Nếu checkpoint tồn tại, hãy tải mô hình từ checkpoint
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = LEARN_pl.load_from_checkpoint(checkpoint_path)  # Tải mô hình từ checkpoint
    else:
        model = LEARN_pl(n_iterations=n_iterations, num_view=num_view, num_detectors=num_detectors)

    dm = CTDataModule(
        data_dir=path_dir,
        batch_size=batch_size,
        num_view=num_view,
        input_size=input_size,
        num_select=-1,
        setting=setting,
        poission_level=poission_level,
    )

    trainer = pl.Trainer(
        accelerator="gpu",  # Sử dụng GPU
        devices=[5],        # Sử dụng 1 GPU
        max_epochs=39,
        logger=tb_logger,
        enable_checkpointing=True,
        callbacks=load_callbacks(n_iter, n_view, noise),
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()