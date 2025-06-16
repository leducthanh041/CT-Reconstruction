# RegFormer_main.py
# RegFormer_main.py (đầu file)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import pytorch_lightning as pl

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
torch.cuda.empty_cache()
from CTSlice_Provider import CTSlice_Provider
from RegFormer_model import RegFormer_pl
from datamodule import CTDataModule
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
    output_path = "/home/thanhld/CT_Reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_" + str(n_iter) + "_iters_bs_1_view_" + str(n_view) + "_noise_" + str(noise) + "_transform/"
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
poission_level = 0

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0_transform"
path_dir = "/home/thanhld/CT_Reconstruction/split/"

n_iterations = 10
batch_size = 1

n_iter, n_view, noise = n_iterations, num_view, str(poission_level)
print("n_iter, n_view, noise", n_iter, n_view, noise)

seed_everything(42, workers=True)
# tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
tb_logger = pl.loggers.TensorBoardLogger("RegFormer_Training_all")


# model = LEARN_pl(n_iterations=n_iterations, num_view=num_view, num_detectors=num_detectors)
options = {
    'views': num_view,
    'dets': num_detectors,
    'width': input_size,
    'height': input_size,
    'dImg': 0.006641*2,       # (giá trị mẫu, điều chỉnh nếu cần)
    'dDet': 0.012858*2,       # (giá trị mẫu)
    'dAng': 0.006134*16, # ví dụ: 1° ở đơn vị radian ~0.01745
    's2r': 5.95,      # giá trị mẫu, tùy theo hệ thống quang học
    'd2r': 4.906,      # giá trị mẫu
    'binshift': 0
}

dm = CTDataModule(data_dir=path_dir, 
                    batch_size=batch_size, 
                    num_view=num_view, 
                    input_size=input_size, 
                    num_select=-1,
                    setting=setting,
                    poission_level=poission_level)

dm.setup(stage="fit")

# Phải gọi setup trước để dm.ct_train được tạo
radon_op, iradon_op, fbp_op, op_norm = dm.ct_train._radon_transform(num_view=num_view)

model = RegFormer_pl(
    block_num=n_iterations,
    options=options,
    radon_op=radon_op,
    fbp_op=fbp_op
)

trainer = pl.Trainer(
    accelerator='gpu',         # Sử dụng GPU
    devices=1,                 # Sử dụng 1 GPU
    max_epochs=50,
    logger=tb_logger,
    enable_checkpointing=True,
    callbacks=load_callbacks(n_iter, n_view, noise)
)

trainer.fit(model, dm)