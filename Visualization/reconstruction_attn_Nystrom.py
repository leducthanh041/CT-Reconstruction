import pytorch_lightning as pl
from models.LEARN_Nys2 import LEARN_Nys_pl
from modules.reconstructor import reconstructor, reconstructor_loss
import torch
import torch.nn as nn
from CTSlice_Provider import CTSlice_Provider
# from CTSlice_Provider_dl import CTSlice_Provider
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import Module
import numpy as np
import cv2
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio
# Tạo mảnh trống bên phải để đặt colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as mtransforms
import torch.nn.functional as F

torch.cuda.empty_cache()
#region --- 5. Hàm tiện ích chuyển numpy/tensor về batch tensor ---

# Hàm hiển thị PSNR/SSIM trên góc phải ảnh reconstructed
def add_psnr_ssim_text(ax, psnr, ssim):
    ax.text(
        0.96, 0.96,
        f'PSNR/SSIM: {psnr:.2f}/{ssim:.2f}',
        color='white',
        fontsize=10,
        ha='right',
        va='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )

def to_batch_tensor(x, device):
    t = x.float() if torch.is_tensor(x) else torch.from_numpy(x).float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.unsqueeze(0).to(device)

#endregion

# --- 6. Chuẩn bị device, dataset, model ---
num_view = 64
input_size = 256
poission_level=0
folder = "AAPM"
folder_V="Nystrom"

print(f'num_view {num_view} and poission_level {poission_level}')
device= torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print("Using device:", device)

path_dir = "/data/uittogether/Thanhld/split/"
transform = transforms.Compose([transforms.Resize(input_size)])

dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)

model = LEARN_Nys_pl.load_from_checkpoint("/data/uittogether/Thanhld/CT-Reconstruction/LEARN_Nystromformer/saved_results_noise_2_with_Nystromformer/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=46.7756.ckpt",
                                        map_location = device)
model.to(device).eval()
                           
# Khởi tạo các biến để lưu ảnh tốt nhất
best_psnr = -float('inf')  # Bắt đầu với giá trị thấp nhất
best_ssim = -float('inf')  # Bắt đầu với giá trị thấp nhất
best_image_index = -1  # Chỉ số ảnh có PSNR và SSIM cao nhất

ssim_metric = StructuralSimilarityIndexMeasure().to(device)
psnr_metric = PeakSignalNoiseRatio().to(device)

# Duyệt qua tất cả các ảnh trong dataset để tìm ảnh có PSNR và SSIM cao nhất
for i in range(len(dataset)):
    phantom_raw, fbp_u_raw, sino_raw = dataset[i]
    phantom = to_batch_tensor(phantom_raw, device)
    fbp_u = to_batch_tensor(fbp_u_raw, device)
    sino = to_batch_tensor(sino_raw, device)

    # Tính toán đầu ra từ mô hình
    with torch.no_grad():
        y_hat, _, _ = model(fbp_u, sino)

    # Tính toán PSNR và SSIM cho ảnh hiện tại
    ssim_value = ssim_metric(y_hat, phantom).item()
    psnr_value = psnr_metric(y_hat, phantom).item()

    # Kiểm tra nếu PSNR và SSIM hiện tại cao hơn ảnh tốt nhất
    if psnr_value > best_psnr and ssim_value > best_ssim:
        best_psnr = psnr_value
        best_ssim = ssim_value
        best_image_index = i  # Cập nhật chỉ số ảnh tốt nhất
    print(f'idx = {i}')

# In ra chỉ số ảnh tốt nhất và giá trị PSNR/SSIM của nó
print(f"Best image index: {best_image_index}")
print(f"Best PSNR: {best_psnr:.2f}, Best SSIM: {best_ssim:.4f}")

# Lấy ảnh có PSNR/SSIM cao nhất
phantom_raw, fbp_u_raw, sino_raw = dataset[best_image_index]
phantom = to_batch_tensor(phantom_raw, device)
fbp_u = to_batch_tensor(fbp_u_raw, device)
sino = to_batch_tensor(sino_raw, device)

# Đưa ảnh tốt nhất vào mô hình
model.to(device).eval()
with torch.no_grad():
    y_hat, outputs, attn_maps = model(fbp_u, sino)

# Tính PSNR/SSIM cho ảnh tốt nhất sau khi inference
ssim_best = ssim_metric(y_hat, phantom).item()
psnr_best = psnr_metric(y_hat, phantom).item()

print(f"After inference, PSNR: {psnr_best:.2f}, SSIM: {ssim_best:.4f}")

    
print(f"Đã thu thập {len(attn_maps)} attention maps")

# --- 8. Visualize ---

def merge_attention_matrices_list(attn_maps_list):
    merged_list = []
    for attn_tuple in attn_maps_list:
        # Lấy các ma trận attn1, attn2_inv, attn3 từ tuple
        attn1, attn2_inv, attn3 = attn_tuple

        # Tiến hành nhân các ma trận với nhau bằng matmul
        temp = torch.matmul(attn1, attn2_inv)
        final_attn = torch.matmul(temp, attn3)

        merged_list.append(final_attn)

    return merged_list


num_iters = len(attn_maps)
attn_maps = merge_attention_matrices_list(attn_maps)
# print('attn_maps:', attn_maps.shape)
N = attn_maps[0].shape[-1]
print('N:', N)
side = int(np.sqrt(N))  # giả định N là số token vuông, để reshape heatmap

# Giữ nguyên 8 cột hình, thêm 1 cột nữa (cột 9) cho colorbar
fig, axes = plt.subplots(2, 9, figsize=(24, 6), gridspec_kw={'width_ratios':[1.2]*8 + [0.05]})
plt.subplots_adjust(
  wspace=0.01,   # khoảng cách ngang
  hspace=0.01    # khoảng cách dọc
)
# Hiển thị Ground Truth (ảnh gốc)
axes[0, 0].imshow(phantom.cpu().squeeze(), cmap='gray')
axes[0, 0].set_title("Ground Truth", fontsize=18)
axes[0, 0].axis("off")

# Hàm vẽ overlay attention heatmap lên ảnh tái tạo
def plot_attention_overlay(ax, image, attention_heatmap, alpha=1):
    # Chuẩn hóa ảnh tái tạo về [0,1]
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Chuẩn hóa heatmap về [0,1]
    heatmap_norm = (attention_heatmap - attention_heatmap.min()) / (attention_heatmap.max() - attention_heatmap.min() + 1e-8)
    
    # Resize heatmap về đúng kích thước ảnh
    h_img, w_img = img_norm.shape
    h_heat, w_heat = heatmap_norm.shape
    if (h_img != h_heat) or (w_img != w_heat):
        heatmap_resized = cv2.resize(heatmap_norm, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap_norm
    
    ax.imshow(img_norm, cmap='gray')
    ax.imshow(heatmap_resized, cmap='viridis', alpha=alpha)
    ax.axis('off')

# Dòng 1: iter 1 đến 7
for i in range(7):
    attn = attn_maps[i][0].cpu().numpy()       # ma trận attention shape (N, N)
    # print(f"Shape attn: {attn.shape}")
    attn_mean = attn.mean(axis=0)  # trung bình qua batch và heads -> (N, N)
    # print(f"Shape attn_mean: {attn_mean.shape}")
    first_row = attn_mean[0, :] 
    # print(f"Shape first_row: {first_row.shape}")
    heatmap = first_row.reshape(side, side)

    # Ảnh tái tạo iteration i (lấy từ output model hoặc bạn cần lưu từng bước khi chạy)
    # Giả sử bạn đã lưu output từng iteration vào list outputs
    # Nếu chưa, bạn cần code thêm phần đó (mình sẽ giúp nếu bạn muốn)
    output_img = outputs[i]  # numpy array shape (256,256) hoặc tương đương

    ax = axes[0, i + 1]
    plot_attention_overlay(ax, output_img, heatmap, alpha=1)
    ax.set_title(f"Iteration {i + 1}", fontsize=18)
    print(f"Done Dòng 1 Iter {i + 1}")

# Dòng 2: iter 8 đến 14
for i in range(7, 14):
    attn = attn_maps[i][0].cpu().numpy()
    attn_mean = attn.mean(axis=0)  # trung bình qua batch và heads -> (N, N)
    first_row = attn_mean[0, :] 
    heatmap = first_row.reshape(side, side)
    output_img = outputs[i]

    ax = axes[1, i - 7]
    plot_attention_overlay(ax, output_img, heatmap, alpha=1)
    ax.set_title(f"Iteration {i + 1}", fontsize=18)
    print(f"Done Dòng 2 Iter {i + 1}")

# Ô cuối cùng: ảnh tái tạo cuối cùng (không overlay)
ax_recon = axes[1, 7]
recon_img = outputs[-1]
ax_recon.imshow(recon_img, cmap='gray')
ax_recon.axis('off')
ax_recon.set_title("Reconstructed", fontsize=18)

add_psnr_ssim_text(ax_recon, psnr_best, ssim_best)
# Ẩn trục của cột thứ 9 (colorbar axes)
axes[0, 8].axis('off')
axes[1, 8].axis('off')

# Lấy bbox 2 dòng 8 cột (không tính cột 9)
bboxes = [ax.get_position() for ax in axes[0, :8]] + [ax.get_position() for ax in axes[1, :8]]
x0 = min(b.x0 for b in bboxes)
y0 = min(b.y0 for b in bboxes)
x1 = max(b.x1 for b in bboxes)
y1 = max(b.y1 for b in bboxes)
full_bbox = mtransforms.Bbox.from_extents(x0, y0, x1, y1)

# Vị trí cột thứ 9 nằm sát mép phải vùng subplot
left = full_bbox.x1 + 0.005
bottom = full_bbox.y0
width_cbar = axes[0,8].get_position().width  # hoặc 0.015
height_cbar = full_bbox.height

# Tạo trục colorbar cao bằng 2 dòng, nằm cột 9
cbar_ax = fig.add_axes([left, bottom, width_cbar + 0.005, height_cbar])

# Lấy heatmap đại diện đã vẽ (ví dụ hình iteration 14 ở axes[1,6])
im = axes[1, 6].images[-1]

# Vẽ colorbar vào trục cbar_ax
im = ax.imshow(heatmap, cmap='viridis')
fig.colorbar(im, cax=cbar_ax)

# Tiếp tục code vẽ, lưu ảnh

plt.tight_layout(rect=[0, 0, left - 0.005, 1])  # tránh trùng với colorbar
plt.show()
plt.savefig(f"./Visualization/{folder_V}/{folder}/{num_view}_view_{poission_level}_noise_{folder}.png", dpi=300)
plt.close()