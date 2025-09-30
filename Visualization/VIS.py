import pytorch_lightning as pl
from models.LEARN_Nys import LEARN_Nys_pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider_dl import CTSlice_Provider
from datamodule import CTDataModule
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import odl
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio

num_view = 32
input_size = 256
poission_level=5e5
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
path_dir = "/home/doanhbc/q3_ThayKhang/CT-reconstruction/split_dl/"
#torch.cuda.empty_cache()

transform = transforms.Compose([transforms.Resize(input_size)])

dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)
print(len(dataset))

phantom, fbp, sino = dataset[8]
print(f'phantom shape: {phantom.shape}')
print(f'fbp shape: {fbp.shape}')
print(f'sino shape: {sino.shape}')
initial = torch.rand(1, 256, 256)

model = LEARN_Nys_pl.load_from_checkpoint(
    "/home/uit2023/LuuTru/Thanhld/Checkpoint/LEARN_Nystromformer/saved_results_noise_2_dl/results_LEARN_14_iters_bs_1_view_32_noise_500000.0_transform_2_9M/epoch=45-val_psnr=39.1148.ckpt",
    map_location=device)
model.eval().to(device)
y_hat = model(fbp.unsqueeze(0), sino.unsqueeze(0))
y_hat = y_hat.squeeze(0)
ssim = StructuralSimilarityIndexMeasure()
ssim_p = ssim(y_hat.unsqueeze(0), phantom.unsqueeze(0))
print(ssim_p)

# Plotting the images
plt.figure(1,figsize=(15,80))
plt.imshow(fbp.permute(1,2,0), cmap='gray')
# plt.title('Initialization')
plt.axis('off')
plt.savefig(f'./Initialization.png', bbox_inches='tight')
plt.close()

plt.figure(1,figsize=(15,80))
plt.subplot(1, 3, 1)
plt.imshow(sino.permute(1,2,0), cmap='gray')
# plt.title('Downsampled\n sinogram')
plt.axis('off')
plt.savefig(f'./Sinogram.png', bbox_inches='tight')
plt.close()

plt.figure(1,figsize=(15,80))
plt.subplot(1, 3, 2)
plt.imshow(y_hat.detach().permute(1,2,0), cmap='gray')
# plt.title('Reconstructed Scan SSIM: {:.02f}'.format(ssim_p.detach().item()))
plt.axis('off')
plt.savefig(f'./Reconstructed.png', bbox_inches='tight')
plt.close()

plt.figure(1,figsize=(15,80))
plt.subplot(1, 3, 3)
plt.imshow(phantom.permute(1,2,0).detach(), cmap='gray')
# plt.title('Reference Scan\n 2304 views')
plt.axis('off')
plt.show()
plt.savefig(f'./Reference_Scan.png', bbox_inches='tight')
