import pytorch_lightning as pl
from models.LEARN import LEARN_pl
from models.LEARN_LongNet import LEARN_LongNet_pl
from models.LEARN_Nys import LEARN_Nys_pl
from models.LEARN_Long import LEARN_Long_pl
from models.RegFormer import RegFormer_pl
from modules.reconstructor import reconstructor, reconstructor_loss
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch
from CTSlice_Provider import CTSlice_Provider
from datamodule import CTDataModule
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import odl
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# region: Define function
def to_batch_tensor(x):
    if torch.is_tensor(x):
        t = x.float()
    else:
        t = torch.from_numpy(x).float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.unsqueeze(0)

def add_psnr_ssim_text(ax, psnr, ssim):
    ax.text(
        0.96, 0.96,
        f'PSNR/SSIM: {psnr:.2f}/{ssim:.2f}',
        color='white',
        fontsize=8,
        ha='right',
        va='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )

def add_inset_zoom(ax, arr, box, zoom_ratio=0.3):
    x, y, w, h = box
    zoomed_img = arr[y:y+h, x:x+w]
    axins = inset_axes(
        ax,
        width=f"{int(zoom_ratio*100)}%",
        height=f"{int(zoom_ratio*100)}%",
        loc='lower left',
        borderpad=0,
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes
    )
    axins.imshow(zoomed_img, cmap='gray')
    axins.set_xticks([])
    axins.set_yticks([])
    zoom_rect = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
    axins.add_patch(zoom_rect)
# endregion

# === Main code ===
num_view = 64
input_size = 256

path_dir = "/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction-main/split/"
transform = transforms.Compose([transforms.Resize(input_size)])

titles = ['Ground truth', 'FBP', 'LEARN', 'LEARN \n LongNet self-attention', 'LEARN \n NystromFormer self-attention', 'LEARN \n LongFormer self-attention', 'RegFormer', 'DuDoTrans']

# region: NOISE 0
poission_level=0
dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)

phantom_raw, fbp_u_raw, sino_raw = dataset[0]
phantom = to_batch_tensor(phantom_raw)
fbp_u = to_batch_tensor(fbp_u_raw)
sino = to_batch_tensor(sino_raw)

model_learn = LEARN_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN/saved_results_noise_2/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=46.2497.ckpt")
model_learn.eval().to('cpu')

model_longnet = LEARN_LongNet_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=44.6984.ckpt")
model_longnet.eval().to('cpu')

model_nys = LEARN_Nys_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_Nystromformer/saved_results_noise_2_with_Nystromformer/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=46.9127.ckpt")
model_nys.eval().to('cpu')

model_long = LEARN_Long_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction-main/LEARN_Long/saved_results_noise_2_with_LongformerAttention/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=09-val_psnr=38.3216.ckpt")
model_long.eval().to('cpu')

radon_op, iradon_op, fbp_op, op_norm = dataset._radon_transform(num_view=num_view)
model_reg = RegFormer_pl.load_from_checkpoint("/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_14_iters_bs_1_view_64_noise_0_transform/epoch=49-val_psnr=48.8738.ckpt",
                                            radon_op=radon_op, fbp_op=fbp_op)
model_reg.eval().to('cpu')

dataset_ddt = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)
# Khởi tạo mô hình reconstructor (theo đúng code train)
model_ddt = reconstructor(dataset_ddt)
model_ddt.eval()

# Load checkpoint
ckpt_path = "/data/uittogether/LuuTru/Thanhld/Sparse-view-CT-reconstruction/DuDoTrans/results/models_"+str(num_view)+"_view_"+str(poission_level)+"_noise/epoch_48iter1919.pth.tar"
checkpoint = torch.load(ckpt_path)

model_ddt.load_state_dict(checkpoint['reconstructor_state'])
model_ddt.eval().to('cpu')
print("Loaded DuDoTrans model from checkpoint.")

device = next(model_learn.parameters()).device

ssim_metric = StructuralSimilarityIndexMeasure().to(device)
psnr_metric = PeakSignalNoiseRatio().to(device)

phantom, fbp_u, sino = phantom.to(device), fbp_u.to(device), sino.to(device)

with torch.no_grad():
    y_hat_learn = model_learn(fbp_u, sino)
    y_hat_longnet = model_longnet(fbp_u, sino)
    y_hat_nys = model_nys(fbp_u, sino)
    y_hat_long = model_long(fbp_u, sino)
    y_hat_reg = model_reg(fbp_u, sino)
    _, _, _, y_hat_ddt = model_ddt(fbp_u, phantom, sino)

ssim_p_learn = ssim_metric(y_hat_learn, phantom).item()
psnr_p_learn = psnr_metric(y_hat_learn, phantom).item()

ssim_p_longnet = ssim_metric(y_hat_longnet, phantom).item()
psnr_p_longnet = psnr_metric(y_hat_longnet, phantom).item()

ssim_p_nys = ssim_metric(y_hat_nys, phantom).item()
psnr_p_nys = psnr_metric(y_hat_nys, phantom).item()

ssim_p_long = ssim_metric(y_hat_long, phantom).item()
psnr_p_long = psnr_metric(y_hat_long, phantom).item()

ssim_p_reg = ssim_metric(y_hat_reg, phantom).item()
psnr_p_reg = psnr_metric(y_hat_reg, phantom).item()

ssim_p_ddt = ssim_metric(y_hat_ddt, phantom).item()
psnr_p_ddt = psnr_metric(y_hat_ddt, phantom).item()

imgs_0 = [phantom, fbp_u, y_hat_learn, y_hat_longnet, y_hat_nys, y_hat_long, y_hat_reg, y_hat_ddt]
psnr_list_0 = [None, None, psnr_p_learn, psnr_p_longnet, psnr_p_nys, psnr_p_long, psnr_p_reg, psnr_p_ddt]
ssim_list_0 = [None, None, ssim_p_learn, ssim_p_longnet, ssim_p_nys, ssim_p_long, ssim_p_reg, ssim_p_ddt]
# endregion

# Move inputs to model device
device = next(model_learn.parameters()).device
phantom = phantom.to(device)
fbp_u    = fbp_u.to(device)
sino     = sino.to(device)

plt.figure(1,figsize=(20,80))

# region: VISUALIZE
datasets = [
    {  # N_p=0
        'label': '$N_p = 0$',
        'imgs': imgs_0,    
        'titles': titles,  
        'psnr_list': psnr_list_0,
        'ssim_list': ssim_list_0,
    }
]

yellow_boxes1 = [(170, 100, 40, 40)]

fig, axes = plt.subplots(1, 8, figsize=(28, 4))  # 1 hàng, 8 cột
plt.subplots_adjust(hspace=0.02, wspace=0.05)

data = datasets[0]  # bạn chỉ dùng datasets[0] vì bạn chỉ quan tâm noise 0
imgs = data['imgs']
titles = data['titles']
psnr_list = data['psnr_list']
ssim_list = data['ssim_list']

for col_idx in range(8):
    ax = axes[col_idx]
    arr = imgs[col_idx].detach().cpu().squeeze().numpy()
    ax.imshow(arr, cmap='gray')
    ax.set_title(titles[col_idx], fontsize=14)
    ax.axis('off')

    # Vẽ box vàng
    for (x, y, w, h) in yellow_boxes1:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    # Thêm PSNR / SSIM
    if psnr_list[col_idx] is not None and ssim_list[col_idx] is not None:
        add_psnr_ssim_text(ax, psnr_list[col_idx], ssim_list[col_idx])

    # Thêm inset zoom (lấy box đầu tiên)
    add_inset_zoom(ax, arr, yellow_boxes1[0])


plt.savefig('./Visualization/reconstruction_with_inset_zoom.png', bbox_inches='tight')

plt.figure(0)
plt.imshow(fbp_u.detach().cpu().squeeze().numpy(), cmap='bone')
plt.title('Initialization')
plt.axis('off')