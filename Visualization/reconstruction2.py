import pytorch_lightning as pl
from models.LEARN import LEARN_pl
from models.LEARN_LongNet import LEARN_LongNet_pl
from models.LEARN_Nys import LEARN_Nys_pl
from models.RegFormer import RegFormer_pl
from modules.reconstructor import reconstructor, reconstructor_loss
import torch
from CTSlice_Provider import CTSlice_Provider
# from CTSlice_Provider_dl import CTSlice_Provider
# from loaders.load_dataset import CTSlice_Provider
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# === Main code ===
num_view = 64
input_size = 256

path_dir = "/home/doanhbc/q3_ThayKhang/CT-reconstruction/split/"
transform = transforms.Compose([transforms.Resize(input_size)])

titles = ['Ground truth', 'FBP', 'LEARN', 'LEARN \n LongNet self-attention', 'LEARN \n NystromFormer self-attention', 'RegFormer', 'DuDoTrans']

# region: NOISE 0
poission_level=0
dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)

phantom_raw, fbp_u_raw, sino_raw = dataset[4]
phantom = to_batch_tensor(phantom_raw)
fbp_u = to_batch_tensor(fbp_u_raw)
sino = to_batch_tensor(sino_raw)

model_learn = LEARN_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN/saved_results_noise_2/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=46.2497.ckpt")
model_learn.eval().to('cpu')

model_longnet = LEARN_LongNet_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=44.6984.ckpt",
                                                    map_location=torch.device('cpu'))
model_longnet.eval().to('cpu')

model_nys = LEARN_Nys_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_Nystromformer/saved_results_noise_2_with_Nystromformer/results_LEARN_14_iters_bs_1_view_64_noise_0_transform/epoch=45-val_psnr=46.9127.ckpt",
                                            map_location=torch.device('cpu'))
model_nys.eval().to('cpu')

radon_op, iradon_op, fbp_op, op_norm = dataset._radon_transform(num_view=num_view)
model_reg = RegFormer_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_14_iters_bs_1_view_64_noise_0_transform/epoch=49-val_psnr=48.8738.ckpt",
                                            radon_op=radon_op, fbp_op=fbp_op)
model_reg.eval().to('cpu')

dataset_ddt = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)
# Khởi tạo mô hình reconstructor (theo đúng code train)
model_ddt = reconstructor(dataset_ddt)
model_ddt.eval()

# Load checkpoint
ckpt_path = "/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/DuDoTrans/results/models_"+str(num_view)+"_view_"+str(poission_level)+"_noise/epoch_48iter1919.pth.tar"
checkpoint = torch.load(ckpt_path, map_location='cpu')

model_ddt.load_state_dict(checkpoint['reconstructor_state'])
print("Loaded DuDoTrans model from checkpoint.")

device = next(model_learn.parameters()).device
phantom, fbp_u, sino = phantom.to(device), fbp_u.to(device), sino.to(device)

y_hat_learn = model_learn(fbp_u, sino)
psnr_p_learn, ssim_p_learn = 46.0959, 0.9872

y_hat_longnet = model_longnet(fbp_u, sino)
psnr_p_longnet, ssim_p_longnet = 44.4621, 0.9831

y_hat_nys = model_nys(fbp_u, sino)
psnr_p_nys, ssim_p_nys = 46.4035, 0.9883

y_hat_reg = model_reg(fbp_u, sino)
psnr_p_reg, ssim_p_reg = 46.7099, 0.9893

with torch.no_grad():
    _, _, _, y_hat_ddt = model_ddt(fbp_u, phantom, sino)
psnr_p_ddt, ssim_p_ddt = 39.3463, 0.9511

imgs_0 = [phantom, fbp_u, y_hat_learn, y_hat_longnet, y_hat_nys, y_hat_reg, y_hat_ddt]
psnr_list_0 = [None, None, psnr_p_learn, psnr_p_longnet, psnr_p_nys, psnr_p_reg, psnr_p_ddt]
ssim_list_0 = [None, None, ssim_p_learn, ssim_p_longnet, ssim_p_nys, ssim_p_reg, ssim_p_ddt]
# endregion

# region: NOISE 5e5
poission_level=5e5
dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)

phantom_raw, fbp_u_raw, sino_raw = dataset[4]
phantom = to_batch_tensor(phantom_raw)
fbp_u = to_batch_tensor(fbp_u_raw)
sino = to_batch_tensor(sino_raw)

model_learn = LEARN_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN/saved_results_noise_2/results_LEARN_14_iters_bs_1_view_64_noise_500000.0_transform/epoch=45-val_psnr=43.0233.ckpt")
model_learn.eval().to('cpu')

model_longnet = LEARN_LongNet_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_64_noise_500000.0_transform/epoch=45-val_psnr=42.6632.ckpt",
                                                    map_location=torch.device('cpu'))
model_longnet.eval().to('cpu')

model_nys = LEARN_Nys_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_Nystromformer/saved_results_noise_2_with_Nystromformer/results_LEARN_14_iters_bs_1_view_64_noise_500000.0_transform/epoch=45-val_psnr=43.5535.ckpt",
                                            map_location=torch.device('cpu'))
model_nys.eval().to('cpu')

radon_op, iradon_op, fbp_op, op_norm = dataset._radon_transform(num_view=num_view)
model_reg = RegFormer_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_14_iters_bs_1_view_64_noise_500000.0_transform/epoch=48-val_psnr=45.0721.ckpt",
                                            radon_op=radon_op, fbp_op=fbp_op)
model_reg.eval().to('cpu')

dataset_ddt = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)
# Khởi tạo mô hình reconstructor (theo đúng code train)
model_ddt = reconstructor(dataset_ddt)
model_ddt.eval()

# Load checkpoint
ckpt_path = "/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/DuDoTrans/results/models_"+str(num_view)+"_view_"+str(poission_level)+"_noise/epoch_48iter1919.pth.tar"
checkpoint = torch.load(ckpt_path, map_location='cpu')

model_ddt.load_state_dict(checkpoint['reconstructor_state'])
print("Loaded DuDoTrans model from checkpoint.")

device = next(model_learn.parameters()).device
phantom, fbp_u, sino = phantom.to(device), fbp_u.to(device), sino.to(device)

y_hat_learn = model_learn(fbp_u, sino)
psnr_p_learn, ssim_p_learn = 42.4394, 0.9765

y_hat_longnet = model_longnet(fbp_u, sino)
psnr_p_longnet, ssim_p_longnet = 42.0934, 0.9755

y_hat_nys = model_nys(fbp_u, sino)
psnr_p_nys, ssim_p_nys = 42.831, 0.9786

y_hat_reg = model_reg(fbp_u, sino)
psnr_p_reg, ssim_p_reg = 43.6572, 0.9808

with torch.no_grad():
    _, _, _, y_hat_ddt = model_ddt(fbp_u, phantom, sino)
psnr_p_ddt, ssim_p_ddt = 38.9575, 0.9482

imgs_5e5 = [phantom, fbp_u, y_hat_learn, y_hat_longnet, y_hat_nys, y_hat_reg, y_hat_ddt]
psnr_list_5e5 = [None, None, psnr_p_learn, psnr_p_longnet, psnr_p_nys, psnr_p_reg, psnr_p_ddt]
ssim_list_5e5 = [None, None, ssim_p_learn, ssim_p_longnet, ssim_p_nys, ssim_p_reg, ssim_p_ddt]
# endregion

# region: NOISE 1e6
poission_level=1e6
dataset = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)

phantom_raw, fbp_u_raw, sino_raw = dataset[4]
phantom = to_batch_tensor(phantom_raw)
fbp_u = to_batch_tensor(fbp_u_raw)
sino = to_batch_tensor(sino_raw)

model_learn = LEARN_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN/saved_results_noise_2/results_LEARN_14_iters_bs_1_view_64_noise_1000000.0_transform/epoch=45-val_psnr=44.2203.ckpt")
model_learn.eval().to('cpu')

model_longnet = LEARN_LongNet_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_LongNet/saved_results_noise_2_with_LongNet/results_LEARN_14_iters_bs_1_view_64_noise_1000000.0_transform/epoch=45-val_psnr=43.4135.ckpt",
                                                    map_location=torch.device('cpu'))
model_longnet.eval().to('cpu')

model_nys = LEARN_Nys_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/LEARN_Nystromformer/saved_results_noise_2_with_Nystromformer/results_LEARN_14_iters_bs_1_view_64_noise_1000000.0_transform/epoch=45-val_psnr=44.4395.ckpt",
                                            map_location=torch.device('cpu'))
model_nys.eval().to('cpu')

radon_op, iradon_op, fbp_op, op_norm = dataset._radon_transform(num_view=num_view)
model_reg = RegFormer_pl.load_from_checkpoint("/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/RegFormer/saved_results_noise_2/results_RegFormer_14_iters_bs_1_view_64_noise_1000000.0_transform/epoch=49-val_psnr=45.9975.ckpt",
                                            radon_op=radon_op, fbp_op=fbp_op)
model_reg.eval().to('cpu')

dataset_ddt = CTSlice_Provider(base_path=path_dir, setting=f"numview_{num_view}_inputsize_256_noise_0_transform",
                           poission_level=poission_level, num_view=num_view, input_size=input_size,
                           transform=transform, test=True, num_select=-1)
# Khởi tạo mô hình reconstructor (theo đúng code train)
model_ddt = reconstructor(dataset_ddt)
model_ddt.eval()

# Load checkpoint
ckpt_path = "/home/uit2023/LuuTru/Thanhld/Sparse-view-CT-reconstruction/DuDoTrans/results/models_"+str(num_view)+"_view_"+str(poission_level)+"_noise/epoch_48iter1919.pth.tar"
checkpoint = torch.load(ckpt_path, map_location='cpu')

model_ddt.load_state_dict(checkpoint['reconstructor_state'])
print("Loaded DuDoTrans model from checkpoint.")

device = next(model_learn.parameters()).device
phantom, fbp_u, sino = phantom.to(device), fbp_u.to(device), sino.to(device)

y_hat_learn = model_learn(fbp_u, sino)
psnr_p_learn, ssim_p_learn = 43.6929, 0.9804

y_hat_longnet = model_longnet(fbp_u, sino)
psnr_p_longnet, ssim_p_longnet = 42.9296, 0.9784

y_hat_nys = model_nys(fbp_u, sino)
psnr_p_nys, ssim_p_nys = 43.789, 0.9816

y_hat_reg = model_reg(fbp_u, sino)
psnr_p_reg, ssim_p_reg = 44.6645, 0.9836

with torch.no_grad():
    _, _, _, y_hat_ddt = model_ddt(fbp_u, phantom, sino)
psnr_p_ddt, ssim_p_ddt = 39.153, 0.9502

imgs_1e6 = [phantom, fbp_u, y_hat_learn, y_hat_longnet, y_hat_nys, y_hat_reg, y_hat_ddt]
psnr_list_1e6 = [None, None, psnr_p_learn, psnr_p_longnet, psnr_p_nys, psnr_p_reg, psnr_p_ddt]
ssim_list_1e6 = [None, None, ssim_p_learn, ssim_p_longnet, ssim_p_nys, ssim_p_reg, ssim_p_ddt]
# endregion

# region: VISUALIZE
datasets = [
    {  # N_p=0
        'label': '$N_p = 0$',
        'imgs': imgs_0,    
        'titles': titles,  
        'psnr_list': psnr_list_0,
        'ssim_list': ssim_list_0,
    },
    {  # N_p=5e5
        'label': '$N_p = 5 \\times 10^5$',
        'imgs': imgs_5e5,    
        'titles': titles,  
        'psnr_list': psnr_list_5e5,
        'ssim_list': ssim_list_5e5,
    },
    {  # N_p=1e6
        'label': '$N_p = 10^6$',
        'imgs': imgs_1e6,    
        'titles': titles,  
        'psnr_list': psnr_list_1e6,
        'ssim_list': ssim_list_1e6,
    },
]

yellow_boxes = [(140, 150, 40, 40)]
fig, axes = plt.subplots(3, 7, figsize=(28, 12))
plt.subplots_adjust(hspace=0.02, wspace=0.05)

for row_idx, data in enumerate(datasets):
    imgs = data['imgs']
    titles = data['titles']
    psnr_list = data['psnr_list']
    ssim_list = data['ssim_list']

    for col_idx in range(7):
        ax = axes[row_idx, col_idx]
        arr = imgs[col_idx].detach().cpu().squeeze().numpy()
        ax.imshow(arr, cmap='gray')
        if row_idx == 0:
            ax.set_title(titles[col_idx], fontsize=14)
        ax.axis('off')

        for (x, y, w, h) in yellow_boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)

        if psnr_list[col_idx] is not None and ssim_list[col_idx] is not None:
            add_psnr_ssim_text(ax, psnr_list[col_idx], ssim_list[col_idx])

        add_inset_zoom(ax, arr, yellow_boxes[0])

    # Thêm text dọc nhãn hàng bên trái mỗi hàng
    # y_pos = 1 - (row_idx + 0.5) / 3  # vẫn giữ công thức, bạn có thể chỉnh nhẹ nếu muốn
    label_ys = [0.75, 0.50, 0.25]
    fig.text(
        0.115,          # vị trí ngang (gần sát trái)
        label_ys[row_idx],          # vị trí dọc chính giữa hàng
        data['label'],
        va='center',
        ha='center',
        rotation='vertical',
        fontsize=16
    )

plt.savefig('./Visualization/reconstruction_views_'+str(num_view)+'_AAPM.png', bbox_inches='tight')
# endregion