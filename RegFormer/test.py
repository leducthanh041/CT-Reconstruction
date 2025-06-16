import pytorch_lightning as pl
from RegFormer_model import RegFormer_pl
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider import CTSlice_Provider
from datamodule import CTDataModule
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import odl
import numpy as np
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio

#lightning_checkpoint = torch.load("LEARN_Training_all/lightning_logs/version_7/hparams.yaml, map_location=lambda storage, loc: storage")
#hyperparams = lightning_checkpoint["hyper_parameters"]
#print(hyperparams)

num_view = 32
input_size = 256
poission_level = 1e6

# path_dir ="AAPM-Mayo-CT-Challenge/"
'''NEW'''
path_dir = "/home/thanhld/CT_Reconstruction/split/"
'''NEW'''

batch_size = 16
#torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(input_size)])

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0_transform"

dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

# Lấy toán tử Radon và FBP từ data module
radon_op, iradon_op, fbp_op, op_norm = dm.ct_test._radon_transform(num_view=num_view)

# Đường dẫn checkpoint
checkpoint_path = "/home/thanhld/CT_Reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_10_iters_bs_1_view_32_noise_1000000.0_transform/epoch=49-val_psnr=42.1747.ckpt"

# Tải mô hình từ checkpoint và truyền vào các toán tử Radon và FBP
model = RegFormer_pl.load_from_checkpoint(
    checkpoint_path, 
    radon_op=radon_op,   # Truyền toán tử Radon
    fbp_op=fbp_op       # Truyền toán tử FBP
)

trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)