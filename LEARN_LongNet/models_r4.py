from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import odl
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dilated_attention_pytorch.dilated_attention import MultiheadDilatedAttention
from odl.contrib import torch as odl_torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


# ---------------------------------------------------------------------------
# LongNetAttentionBlock
# Tokenize ảnh → apply dilated attention → reshape lại
# ---------------------------------------------------------------------------
class LongNetAttentionBlock(nn.Module):
    """Tokenize input (batch, 48, 256, 256) via 2×2 patches → dilated attention → reconstruct."""

    def __init__(
        self,
        window_size: int = 2,
        patch_channels: int = 48,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.patch_channels = patch_channels
        self.image_size = image_size
        self.new_spatial = image_size // window_size  # 128
        self.token_dim = patch_channels * (window_size**2)  # 192

        # dilation_rates / segment_lengths phải là Sequence[int], không phải int
        self.longnet_attention = MultiheadDilatedAttention(
            embed_dim=self.token_dim,
            num_heads=4,
            dilation_rates=[4],
            segment_lengths=[1024],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokenize → attend → reconstruct.

        Args:
            x: Tensor shape (batch, patch_channels, image_size, image_size).

        Returns:
            Reconstructed tensor cùng shape với input.
        """
        batch_size, channels, height, width = x.shape

        # Tokenize: gom mỗi patch 2×2 với stride=2
        # Kết quả: (batch, 48, 128, 128, 2, 2)
        tokens = (
            x.unfold(2, self.window_size, self.window_size)
            .unfold(3, self.window_size, self.window_size)
        )

        # Reshape: (batch, 48, 128*128, 2*2)
        tokens = tokens.contiguous().view(
            batch_size, channels, -1, self.window_size * self.window_size
        )

        # Permute: (batch, 128*128, 48, 4)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()

        # Flatten: (batch, 16384, 192)
        tokens = tokens.view(batch_size, -1, self.token_dim)

        # Dilated attention
        attention_output = self.longnet_attention(tokens, tokens, tokens)
        # Reshape lại thành ảnh: (batch, 128, 128, 48, 2, 2)
        tokens_reshaped = attention_output[0].view(
            batch_size,
            self.new_spatial,
            self.new_spatial,
            self.patch_channels,
            self.window_size,
            self.window_size,
        )

        # Permute: (batch, 48, 128, 2, 128, 2)
        tokens_reshaped = tokens_reshaped.permute(0, 3, 1, 4, 2, 5).contiguous()

        # Gộp: (batch, 48, 256, 256)
        reconstructed = tokens_reshaped.view(
            batch_size,
            self.patch_channels,
            self.new_spatial * self.window_size,
            self.new_spatial * self.window_size,
        )

        return reconstructed


# ---------------------------------------------------------------------------
# RegularizationBlock
# ---------------------------------------------------------------------------
class RegularizationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5,
        patch_channels: int = 48,
    ) -> None:
        super().__init__()
        padding_value = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)

        # LongNet attention block sau conv1
        self.longnet_attn = LongNetAttentionBlock(
            window_size=2,
            patch_channels=patch_channels,
            image_size=256,
        )

        self.conv2 = nn.Conv2d(
            patch_channels,
            patch_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)

        self.conv3 = nn.Conv2d(
            patch_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_value,
        )
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.longnet_attn(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ---------------------------------------------------------------------------
# GradientFunction
# ---------------------------------------------------------------------------
class GradientFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.regularitation_term = RegularizationBlock()
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        forward_module: nn.Module,
        backward_module: nn.Module,
    ) -> torch.Tensor:
        data_fidelity_term = forward_module(x_t) - y
        bp_data_fidelity = backward_module(data_fidelity_term)
        reg_value = self.regularitation_term(x_t)
        gradient = self.alpha * bp_data_fidelity + reg_value
        return gradient


# ---------------------------------------------------------------------------
# LEARN_pl — PyTorch Lightning module
# ---------------------------------------------------------------------------
class LEARN_pl(pl.LightningModule):
    def __init__(
        self,
        n_iterations: int,
        num_view: int,
        num_detectors: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gradient_list = nn.ModuleList(
            [GradientFunction() for _ in range(n_iterations)]
        )
        self.initial_lr = 1e-4
        self.final_lr = 1e-5
        self.num_iter = n_iterations

        radon_curr, fbp_curr = self.radon_transform(
            num_view=num_view,
            num_detectors=num_detectors,
        )
        self.forward_module = radon_curr
        self.backward_module = fbp_curr

        self.grid: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _get_batch_data_range(target: torch.Tensor) -> Tuple[float, float]:
        """Compute (min, max) của batch để dùng làm data_range cho metrics."""
        batch_min = float(target.amin())
        batch_max = float(target.amax())

        if np.isclose(batch_max, batch_min):
            batch_max = batch_min + 1e-8

        return (batch_min, batch_max)

    @staticmethod
    def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_iter):
            x_t = x_t - self.gradient_list[i](
                x_t, y, self.forward_module, self.backward_module
            )
        return x_t

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5,
            eta_min=self.final_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        phantom, fbp_u, sino_noisy = train_batch
        x_t = fbp_u
        y = sino_noisy

        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    def validation_step(self, val_batch: tuple, batch_idx: int) -> dict:
        phantom, fbp_u, sino_noisy = val_batch
        x_t = fbp_u
        y = sino_noisy

        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)

        ssim_p = structural_similarity_index_measure(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        psnr_p = peak_signal_noise_ratio(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        rmse_p = self.rmse(phantom, x_reconstructed)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ssim", ssim_p, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_p, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse_p, on_step=False, on_epoch=True, prog_bar=False)

        self.grid = torchvision.utils.make_grid(
            x_reconstructed.detach().clamp(min=0.0)
        )

        return {
            "val_loss": loss,
            "val_ssim": ssim_p,
            "val_psnr": psnr_p,
            "val_rmse": rmse_p,
        }

    # -----------------------------------------------------------------------
    # Test
    # -----------------------------------------------------------------------
    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        phantom, fbp_u, sino_noisy = batch
        x_t = fbp_u
        y = sino_noisy

        x_reconstructed = self.forward(x_t, y)
        loss = F.mse_loss(phantom, x_reconstructed)

        data_range = self._get_batch_data_range(phantom)

        ssim_p = structural_similarity_index_measure(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        psnr_p = peak_signal_noise_ratio(
            x_reconstructed,
            phantom,
            data_range=data_range,
        )
        rmse_p = self.rmse(phantom, x_reconstructed)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim_p, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr_p, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse_p, on_step=False, on_epoch=True)

        return {
            "SSIM": ssim_p,
            "PSNR": psnr_p,
            "RMSE": rmse_p,
        }

    # -----------------------------------------------------------------------
    # Log images at epoch end
    # -----------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        if self.grid is not None and self.logger is not None:
            tag = f"generated_images_epoch_{self.current_epoch}"
            self.logger.experiment.add_image(tag, self.grid, self.current_epoch)

    # -----------------------------------------------------------------------
    # Radon / FBP transform via ODL
    # -----------------------------------------------------------------------
    @staticmethod
    def radon_transform(
        num_view: int = 64,
        start_ang: float = 0.0,
        end_ang: float = 2 * np.pi,
        num_detectors: int = 800,
    ) -> Tuple[nn.Module, nn.Module]:
        """Build forward (RayTransform) và backward (FBP) operators."""
        xx = 200
        space = odl.uniform_discr(
            [-xx, -xx],
            [xx, xx],
            [256, 256],
            dtype="float32",
        )

        angle_partition = odl.uniform_partition(start_ang, end_ang, int(num_view))
        detector_partition = odl.uniform_partition(-480, 480, num_detectors)

        geometry = odl.tomo.FanBeamGeometry(
            angle_partition,
            detector_partition,
            src_radius=600,
            det_radius=290,
        )

        operator = odl.tomo.RayTransform(space, geometry, impl="astra_cuda")
        op_layer = odl_torch.operator.OperatorModule(operator)

        fbp = (
            odl.tomo.fbp_op(operator, filter_type="Ram-Lak", frequency_scaling=0.9)
            * np.sqrt(2)
        )
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_fbp
