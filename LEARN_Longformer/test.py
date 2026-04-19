import pytorch_lightning as pl
from models import LEARN_pl
from pytorch_lightning import Trainer
import torch
from datamodule import CTDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

num_view = 18
input_size = 256
poission_level = 0

'''NEW'''
path_dir = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split/"
'''NEW'''

batch_size = 16
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

model = LEARN_pl.load_from_checkpoint("/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_Longformer/saved_results_noise_2_with_Longformer/results_LEARN_14_iters_bs_1_view_18_noise_0_transform/epoch=46-val_psnr=33.7374.ckpt")

setting = "numview_"+str(num_view)+"_inputsize_256_noise_0"

dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',devices=[4],max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

