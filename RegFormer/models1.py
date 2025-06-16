# RegFormer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.utils import make_grid
import odl
from odl.contrib import torch as odl_torch
import numpy as np
import pytorch_lightning as pl
from torch.autograd import Function
import torchvision.models as models
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio
from swin_transformer import BasicLayer
# import ctlib
import swin_transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# ---------------------------
# Phần 1: Các module cơ bản của RegFormer (được gộp từ file model.py)
# ---------------------------

class prj_module(nn.Module):
    def __init__(self, forward_module, backward_module):
        super(prj_module, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.forward_module = forward_module
        self.backward_module = backward_module
        self.weight.data.zero_()
   
    def forward(self, input_data, proj):
        p_tmp = self.forward_module(input_data)
        y_error = proj - p_tmp
        x_error = self.backward_module(y_error)
        return self.weight * x_error + input_data

class ConvBlock(nn.Module):
    def __init__(self, dim, first=False, last=False) -> None:
        super().__init__()
        if first:
            self.conv1 = nn.Conv2d(1, dim, kernel_size=5, padding=2)
        else:
            self.conv1 = nn.Conv2d(1, dim // 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=5, padding=2)        
        if last:
            self.trans_embed = None
        else:
            self.trans_embed = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):        
        x = self.relu(self.conv1(x))
        if y is not None:
            x = torch.cat((x, y), dim=1)
        x = self.relu(self.conv2(x))
        out = self.conv3(x)
        z = self.trans_embed(x) if self.trans_embed is not None else None
        return out, z

class IterBlock(nn.Module):
    def __init__(self, options, idx, prj, last=False):
        super(IterBlock, self).__init__()

        self.block1 = prj

        first = True if idx == 0 else False
        if (idx % 2 == 0):
            self.block2 = ConvBlock(96, first=first, last=last)
        else:
            self.block2 = Transformer(first=first, last=last)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj, z):
        tmp1 = self.block1(input_data, proj)
        tmp2, z_out = self.block2(input_data, z)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output, z_out

class regformer(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(regformer, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        prj = prj_module(kwargs['radon_curr'], kwargs['fbp_curr'])
        options = torch.Tensor([views, dets, width, height, dImg, dDet, 0, dAng, s2r, d2r, binshift, 0])
        self.model = nn.ModuleList([IterBlock(options, i, prj, last=True if i == block_num - 1 else False) for i in range(block_num)])
    
    def forward(self, input_data, proj):
        x = input_data
        z = None
        for index, module in enumerate(self.model):
            x, z = module(x, proj, z)
        return x

# Một Transformer block dựa trên Swin Transformer
class Transformer(nn.Module):
    def __init__(self, img_size=256, embed_dim=96, depths=[2], num_heads=[3],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, first=False, last=False, **kwargs):
        super(Transformer, self).__init__()
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        if not first:
            self.patch_embed = nn.Conv2d(1, embed_dim // 2, 3, 1, 1)
        else:
            self.patch_embed = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.embed_reverse = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        if not last:
            self.trans_embed = nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1)
        else:
            self.trans_embed = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim,
                               input_resolution=(img_size, img_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        # Khởi tạo trọng số theo cách cơ bản
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 0.01)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        if y is not None:
            x = torch.cat((x, y), dim=1)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).transpose(1, 2).view(B, -1, H, W)
        out = self.embed_reverse(x)
        z = self.trans_embed(x) if self.trans_embed is not None else None
        return out, z

# ---------------------------
# Phần 2: Lớp Lightning Module bọc RegFormer (cấu trúc như LEARN)
# ---------------------------
class RegFormer_pl(pl.LightningModule):
    def __init__(self, block_num, options):
        """
        block_num: Số lượng iteration block (ví dụ như số iterations)
        options: dictionary chứa các tham số cho RegFormer:
           'views', 'dets', 'width', 'height', 'dImg', 'dDet', 'dAng', 's2r', 'd2r', 'binshift'
        """
        super(RegFormer_pl, self).__init__()
        self.save_hyperparameters()

        self.initial_lr = 1e-4
        self.final_lr = 1e-5

        self.num_iter = block_num
        self.ssim = StructuralSimilarityIndexMeasure()
        self.psnr = PeakSignalNoiseRatio()

        radon_curr, fbp_curr = self.radon_transform(num_view=options['views'], num_detectors=options['dets'])

        self.regformer = regformer(block_num=block_num, **options, radon_curr=radon_curr, fbp_curr=fbp_curr)

    def forward(self,x_t,y):
        x_t = self.regformer(x_t, y)
        return x_t

    def radon_transform(self, num_view=64, start_ang=0, end_ang=2*np.pi, num_detectors=800):
        # the function is used to generate fp, bp, fbp functions
        # the physical parameters is set as MetaInvNet and EPNet
        xx=200
        space=odl.uniform_discr([-xx, -xx], [xx, xx], [256,256], dtype='float32')
        angles=np.array(num_view).astype(int)
        angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
        detector_partition=odl.uniform_partition(-480, 480, num_detectors)
        geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        #geometry=odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
        operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        #op_norm=odl.operator.power_method_opnorm(operator)
        #op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

        op_layer=odl_torch.operator.OperatorModule(operator)
        #op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
        fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
        op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp

    def training_step(self, train_batch, batch_idx):
        phantom, fbp_u, sino_noisy = train_batch
        x_t = fbp_u # Hình ảnh khởi tạo ban đầu
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda() # Nếu cần dùng làm khởi tạo
        x_reconstructed = self.forward(x_t, y)        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)

        #self.log('train_loss', loss)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step) 
        return loss

    def validation_step(self, val_batch, batch_idx):
        phantom, fbp_u, sino_noisy = val_batch
        x_t = fbp_u
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda()
        x_reconstructed = self.forward(x_t, y)
        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)
        ssim_p = self.ssim(x_reconstructed, phantom)
        
        psnr_p = self.psnr(x_reconstructed, phantom)
        '''NEW''' 
        print(f"PSNR Value: {psnr_p.item()}")  # Debugging output
        rmse_p = self.rmse(x_reconstructed, phantom)
        
        #self.log('val_loss', loss)
        self.logger.experiment.add_scalars('loss', {'validation': loss},self.global_step)
        self.log('val_ssim', ssim_p)
        self.log('val_psnr', psnr_p, on_epoch=True)
        self.log('val_rmse', rmse_p)
        
        self.grid = torchvision.utils.make_grid(x_reconstructed)

    def test_step(self, test_batch, batch_idx):
        phantom, fbp_u, sino_noisy = test_batch
        x_t = fbp_u
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda()
        x_reconstructed = self.forward(x_t, y)
        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)
        ssim_p = self.ssim(x_reconstructed, phantom)
        psnr_p = self.psnr(x_reconstructed, phantom)
        rmse_p = self.rmse(x_reconstructed, phantom)
        
        test_out = {
            "SSIM": ssim_p,
            "PSNR": psnr_p,
            "RMSE": rmse_p
            }
        
        self.log('test_loss', loss)
        self.log('test_ssim', ssim_p)
        self.log('test_psnr', psnr_p)
        self.log('test_rmse', rmse_p)
        return test_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def rmse(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))