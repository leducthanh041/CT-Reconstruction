import pytorch_lightning as pl
from models import LEARN_pl
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider import CTSlice_Provider
from datamodule import CTDataModule
import torch.nn as nn
from models import GradientFunction
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
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio

#lightning_checkpoint = torch.load("LEARN_Training_all/lightning_logs/version_7/hparams.yaml, map_location=lambda storage, loc: storage")
#hyperparams = lightning_checkpoint["hyper_parameters"]
#print(hyperparams)

# num_view = 64
'''NEW'''
num_view = 32
'''NEW'''
input_size = 256
poission_level = 5e5
#poission_level = 1e6
# poission_level = 0

# path_dir ="AAPM-Mayo-CT-Challenge/"
'''NEW'''
path_dir = "/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction-main/split/"
'''NEW'''

batch_size = 16
#torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

#tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction-main/CT_Reconstruction_LEARN_paper/saved_results_noise_2_with_LongformerAttention/results_LEARN_14_iters_bs_1_view_32_noise_500000.0_transform/epoch=00-val_psnr=26.8353.ckpt")

'''NEW'''
setting = "numview_32_inputsize_256_noise_0_transform"
'''NEW'''

# '''NEW'''
# setting = "numview_64_inputsize_256_noise_0_transform"
# '''NEW'''

dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

