import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from models import LEARN_pl
from datamodule_dl import CTDataModule
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os

import numpy as np
from pytorch_lightning import seed_everything

def load_callbacks(n_iter, n_view, noise):

    Mycallbacks = []
    # Make output path
    output_path = "/home/thanhld/CT_Reconstruction/LEARN_Longformer/saved_results_noise_2_dl_with_Longformer/results_LEARN_" + str(n_iter) + "_iters_bs_1_view_" + str(n_view) + "_noise_" + str(noise) + "_transform/"
    os.makedirs(output_path, exist_ok=True)

    early_stop_callback = EarlyStopping(
        monitor='val_psnr', # val_ssim, val_psnr, val_rmse
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)
    Mycallbacks.append(ModelCheckpoint(monitor = 'val_psnr',
                                        dirpath = output_path,
                                        filename = '{epoch:02d}-{val_psnr:.4f}',
                                        verbose = True,
                                        save_last = True,
                                        save_top_k = 1,
                                        mode = 'max',
                                        save_weights_only = True))
    return Mycallbacks

torch.manual_seed(42)
num_view = 64
input_size = 256
num_detectors = 512
poission_level = 5e5

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0_transform"
path_dir = "/home/thanhld/CT_Reconstruction/split_dl/"

n_iterations = 14
batch_size = 1

n_iter, n_view, noise = n_iterations, num_view, str(poission_level)
print("n_iter, n_view, noise", n_iter, n_view, noise)

seed_everything(42, workers=True)
tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
# Đường dẫn tới checkpoint
checkpoint_path = "/home/thanhld/CT_Reconstruction/LEARN_Longformer/saved_results_noise_2_dl_with_Longformer/results_LEARN_14_iters_bs_1_view_64_noise_500000.0_transform/epoch=00-val_psnr=31.4853.ckpt"

# Nếu checkpoint tồn tại, hãy tải mô hình từ checkpoint
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = LEARN_pl.load_from_checkpoint(checkpoint_path)  # Tải mô hình từ checkpoint
else:
    model = LEARN_pl(n_iterations=n_iterations, num_view=num_view, num_detectors=num_detectors)

# model = LEARN_pl(n_iterations=n_iterations, num_view=num_view, num_detectors=num_detectors)

dm = CTDataModule(data_dir=path_dir, 
                    batch_size=batch_size, 
                    num_view=num_view, 
                    input_size=input_size, 
                    num_select=-1,
                    setting=setting,
                    poission_level=poission_level)


# Giả sử model của bạn là một đối tượng của lớp model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

trainer = pl.Trainer(
    accelerator='gpu',         # Sử dụng GPU
    devices=[2],                 # Sử dụng 1 GPU
    max_epochs=49,
    logger=tb_logger,
    enable_checkpointing=True,
    callbacks=load_callbacks(n_iter, n_view, noise)
)


trainer.fit(model, dm)