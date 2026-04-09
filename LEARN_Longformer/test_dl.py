from models import LEARN_pl
from pytorch_lightning import Trainer
from datamodule_dl import CTDataModule
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl

num_view = 128
input_size = 256
poission_level = 0

path_dir = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split_dl/"

batch_size = 16
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

model = LEARN_pl.load_from_checkpoint("/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_Longformer/saved_results_noise_2_dl/results_LEARN_14_iters_bs_1_view_128_noise_0_transform/epoch=02-val_psnr=42.9921.ckpt",
                                    map_location='cuda')

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0"


dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',devices=[1],max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

