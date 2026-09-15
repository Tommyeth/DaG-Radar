import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianBEVFeatureFusion(nn.Module):
    """Align and project dense/Gaussian BEV features for the fusion control."""

    def __init__(self, dense_channels, gaussian_channels, out_channels):
        super().__init__()
        self.dense_channels = int(dense_channels)
        self.gaussian_channels = int(gaussian_channels)
        self.out_channels = int(out_channels)
        self.projection = nn.Conv2d(
            self.dense_channels + self.gaussian_channels,
            self.out_channels,
            kernel_size=1,
        )

    def forward(self, dense_bev, gaussian_bev):
        if dense_bev.ndim != 4 or gaussian_bev.ndim != 4:
            raise ValueError("feature fusion expects BCHW tensors")
        if dense_bev.shape[0] != gaussian_bev.shape[0]:
            raise ValueError("dense and Gaussian batch size must match")
        if dense_bev.shape[1] != self.dense_channels:
            raise ValueError(
                "unexpected dense BEV channel count: "
                f"expected {self.dense_channels}, got {dense_bev.shape[1]}"
            )
        if gaussian_bev.shape[1] != self.gaussian_channels:
            raise ValueError(
                "unexpected Gaussian BEV channel count: "
                f"expected {self.gaussian_channels}, got {gaussian_bev.shape[1]}"
            )
        if gaussian_bev.shape[-2:] != dense_bev.shape[-2:]:
            gaussian_bev = F.interpolate(
                gaussian_bev,
                size=dense_bev.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.projection(torch.cat((dense_bev, gaussian_bev), dim=1))
