import pytorch_lightning as pl
from models import LEARN_pl
from pytorch_lightning import Trainer
from datamodule_dl import CTDataModule
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

#lightning_checkpoint = torch.load("LEARN_Training_all/lightning_logs/version_7/hparams.yaml, map_location=lambda storage, loc: storage")
#hyperparams = lightning_checkpoint["hyper_parameters"]
#print(hyperparams)

num_view = 128
input_size = 256
poission_level = 5e5

# path_dir ="AAPM-Mayo-CT-Challenge/"
'''NEW'''
path_dir = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split_dl/"
'''NEW'''

batch_size = 16
#torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

#tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl.load_from_checkpoint("/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_LongNet/saved_results_noise_2_dl/results_LEARN_14_iters_bs_1_view_128_noise_500000.0_transform/epoch=46-val_psnr=43.7065.ckpt")

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0"


dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',devices=[4],max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

