from models2_9M import LEARN_pl
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from datamodule_dl import CTDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

num_view = 18
input_size = 256
poission_level = 1e6

'''NEW'''
path_dir = "/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/split_dl/"
'''NEW'''

batch_size = 16
#torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(input_size)])
n_iterations = 10

#tb_logger = pl.loggers.TensorBoardLogger("LEARN_Training_all")
model = LEARN_pl.load_from_checkpoint("/mmlab_students/storageStudents/nguyenvd/Thanhld/CT-Reconstruction/LEARN_Nystromformer/saved_results_noise_2_dl/results_LEARN_14_iters_bs_1_view_18_noise_1000000.0_transform/epoch=37-val_psnr=32.2222.ckpt")

'''NEW'''
setting = "numview_"+str(num_view)+"_inputsize_256_noise_0"
'''NEW'''
dm = CTDataModule(data_dir=path_dir, batch_size=batch_size, num_view=num_view, input_size=input_size,
                    setting=setting, poission_level=poission_level)
dm.setup('test')

trainer = pl.Trainer(accelerator='gpu',devices=[1],max_epochs=10,enable_checkpointing=True)
trainer.test(model, dataloaders=dm)
#test_model(model, test_loader)

