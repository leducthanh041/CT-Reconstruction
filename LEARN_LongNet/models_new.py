from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import astra
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics.image import StructuralSimilarityIndexMeasure 
from torchmetrics.image import PeakSignalNoiseRatio
import odl
import numpy as np
from odl.contrib import torch as odl_torch
from long_net import DilatedAttention

# ----------------------------
# Lớp LongNetAttentionBlock: áp dụng attention lên output của Conv (tokenization, attention, expand lại)
# ----------------------------
class LongNetAttentionBlock(nn.Module):
    """
    Nhận input với shape (batch, 48, 256, 256), sau đó:
      - Gom mỗi patch 2x2 thành 1 token (token_dim = 48*2*2 = 192)
      - Tạo sequence các token có shape (batch, 16384, 192)
      - Áp dụng LongNet attention (với attention_window = 2)
      - Reshape lại về tensor có shape (batch, 48, 256, 256)
    """
    def __init__(self, window_size=2, patch_channels=48, image_size=256):
        super(LongNetAttentionBlock, self).__init__()
        self.window_size = window_size
        self.patch_channels = patch_channels  # 48
        self.image_size = image_size
        self.new_spatial = image_size // window_size  # 256/2 = 128
        self.token_dim = patch_channels * (window_size ** 2)  # 48*4 = 192

                
        # Chỉ khởi tạo lớp attention đầu tiên
        self.longnet_attention = DilatedAttention(dim=self.token_dim, heads=4, dilation_rate=1, segment_size=16384 // 4, dropout=0.1, qk_norm=True)

    def forward(self, x, global_attention_indices=None):
        # x: (batch, 48, 256, 256)
        batch_size, channels, height, width = x.shape

        # Tokenize: gom mỗi patch 2x2 với stride = 2
        # Kết quả: (batch, 48, 128, 128, 2, 2)
        tokens = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)

        # Reshape: (batch, 48, 128*128, 2*2)
        tokens = tokens.contiguous().view(batch_size, channels, -1, self.window_size * self.window_size)

        # Permute: (batch, 128*128, 48, 4)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()

        # Flatten: (batch, 16384, 48*4=192)
        tokens = tokens.view(batch_size, -1, self.token_dim)        
       
        # Áp dụng lớp attention của LongNet
        # print(f"Shape before attention: {tokens.shape}")
        attention_output = self.longnet_attention(tokens)
        # print(f"Shape after attention: {attention_output.shape}")
        

        # Expand lại: reshape sequence thành tensor hình ảnh
        # Reshape thành: (batch, 128, 128, 48, 2, 2)
        tokens_reshaped = attention_output.view(batch_size, self.new_spatial, self.new_spatial, self.patch_channels, self.window_size, self.window_size)

        # Permute: (batch, 48, 128, 2, 128, 2)
        tokens_reshaped = tokens_reshaped.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Gộp các chiều không gian: (batch, 48, 128*2, 128*2) = (batch, 48, 256, 256)
        reconstructed = tokens_reshaped.view(batch_size, self.patch_channels, self.new_spatial * self.window_size, self.new_spatial * self.window_size)

        return reconstructed

class RegularizationBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5):
        super(RegularizationBlock, self).__init__()
        padding_value = kernel_size // 2  # Với kernel_size=5 thì padding = 2
    
        self.conv1 = nn.Conv2d(in_channels, 48, kernel_size=kernel_size, padding=padding_value)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        '''
        NEW
        '''
        # Lớp LongNet attention block được áp dụng sau Conv1
        self.longnet_attn = LongNetAttentionBlock(window_size=2, patch_channels=48, image_size=256)
        '''
        NEW
        '''
    
        self.conv2 = nn.Conv2d(48, 48, kernel_size=kernel_size, padding=padding_value)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
    
        self.conv3 = nn.Conv2d(48, out_channels, kernel_size=kernel_size, padding=padding_value)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.longnet_attn(x)  # Attention: token hóa -> attention -> expand lại -> (batch, 48, 256, 256)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class GradientFunction(nn.Module):
    
    def __init__(self):
        super(GradientFunction, self).__init__()
        self.regularitation_term = RegularizationBlock()
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self,x_t, y, forward_module, backward_module):
        data_fidelity_term = forward_module(x_t) - y
        bp_data_fidelity = backward_module(data_fidelity_term)
        reg_value = self.regularitation_term(x_t)
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient

class LEARN_pl(pl.LightningModule):
    def __init__(self, n_iterations,num_view, num_detectors):
        super(LEARN_pl, self).__init__()
        self.save_hyperparameters()
        self.gradient_list = nn.ModuleList([GradientFunction() for _ in range(n_iterations)])
        self.initial_lr = 1e-4
        self.final_lr = 1e-5
        self.num_iter = n_iterations
        self.ssim = StructuralSimilarityIndexMeasure()
        self.psnr = PeakSignalNoiseRatio()
        #self.rmse = RootMeanSquaredErrorUsingSlidingWindow()
        #radon_curr, fbp_curr = radon(num_view=num_view)
        radon_curr, fbp_curr = self.radon_transform(num_view=num_view, num_detectors=num_detectors)
        
        self.forward_module = radon_curr
        self.backward_module = fbp_curr
        
        self.grid = None
    
    def forward(self,x_t,y):
        # x_t là hình ảnh ban đầu (hoặc hình ảnh được cập nhật)
        x_t = x_t
        for i in range(self.num_iter):
            x_t = x_t - self.gradient_list[i](x_t, y, self.forward_module, self.backward_module)
        return x_t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.final_lr)
        
        return (
        {   "optimizer":optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }   )

    def training_step(self, train_batch, batch_idx):
        phantom, fbp_u, sino_noisy = train_batch
        x_t = fbp_u # Hình ảnh khởi tạo ban đầu
        y = sino_noisy
        initial = torch.rand(y.shape[0],1,256, 256).cuda() # Nếu cần dùng làm khởi tạo
        x_reconstructed = self.forward(x_t, y)        
        loss = nn.functional.mse_loss(phantom, x_reconstructed)

        torch.cuda.empty_cache()

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
        
        torch.cuda.empty_cache()

        #self.log('val_loss', loss)
        self.logger.experiment.add_scalars('loss', {'validation': loss},self.global_step)
        self.log('val_ssim', ssim_p)
        self.log('val_psnr', psnr_p, on_epoch=True)
        self.log('val_rmse', rmse_p)
        
        self.grid = torchvision.utils.make_grid(x_reconstructed)
        
    def test_step(self, batch, batch_idx):
        phantom, fbp_u, sino_noisy = batch
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
    
    def on_validation_epoch_end(self):
        self.logger.experiment.add_image("generated_images", self.grid, self.current_epoch,)
    
    
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
    
    def on_validation_epoch_end(self):
        tag = f"generated_images_epoch_{self.current_epoch}"
        self.logger.experiment.add_image(tag, self.grid, self.current_epoch)
        
    def rmse(self,y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))