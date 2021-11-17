import torch
from torch import nn as nn
from torch.nn import functional as F
import math

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
import cv2
import os
import numpy as np
class DIM:
    BATCH = 0
    CHANNEL = 1
    WIDTH = 2
    HEIGHT = 3
@ARCH_REGISTRY.register()
class UpCunet(nn.Module):
    def __init__(self, channels=3):
        super(UpCunet, self).__init__()
        self.cunet_unet1 = CunetUnet1(channels, deconv=True)#True:2x-16#False:1x-8
        self.cunet_unet2 = CunetUnet2(channels, deconv=False)#-20
        self.spatial_zero_padding = SpatialZeroPadding(-20)

    def forward(self, x):
        # x = np.pad(x, [(18, 18 ), (18, 18), (0, 0)], mode='reflect')#训练不支持奇数，反正切好了偶数的
        x=F.pad(x,(18,18,18,18),'reflect')
        x = self.cunet_unet1.forward(x)
        x0 = self.cunet_unet2.forward(x)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        # x = torch.clamp(x, min=0, max=1)
        return x

class CunetUnet1(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(64, 128, 64, se=True)
        self.unet_branch = UnetBranch(block1, 64, 64, depad=-4)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)

    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class CunetUnet2(nn.Module):
    def __init__(self, channels: int, deconv: bool):
        super().__init__()
        self.unet_conv = UnetConv(channels, 32, 64, se=False)
        block1 = UnetConv(128, 256, 128, se=True)
        block2 = nn.Sequential(
            UnetConv(64, 64, 128, se=True),
            UnetBranch(block1, 128, 128, depad=-4),
            UnetConv(128, 64, 64, se=True),
        )
        self.unet_branch = UnetBranch(block2, 64, 64, depad=-16)
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.1)
        if deconv:
            # Uncertain
            self.conv1 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(64, channels, kernel_size=3)

    def forward(self, x):
        x = self.unet_conv(x)
        x = self.unet_branch(x)
        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        return x


class UnetConv(nn.Module):
    def __init__(self, channels_in: int, channels_mid: int, channels_out: int, se: bool):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_mid, 3)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(channels_mid, channels_out, 3)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.se = se
        if se:
            self.se_block = SEBlock(channels_out, r=8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        if self.se:
            x = self.se_block(x)
        return x


class UnetBranch(nn.Module):
    def __init__(self, insert: nn.Module, channels_in: int, channels_out: int, depad: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels_in, channels_in, kernel_size=2, stride=2)
        self.lrelu0 = nn.LeakyReLU(0.1)
        self.insert = insert
        self.conv1 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=2, stride=2)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.spatial_zero_padding = SpatialZeroPadding(depad)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.lrelu0(x0)
        x0 = self.insert(x0)
        x0 = self.conv1(x0)
        x0 = self.lrelu1(x0)
        x1 = self.spatial_zero_padding(x)
        x = torch.add(x0, x1)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels_out: int, r: int):
        super().__init__()
        channels_mid = math.floor(channels_out / r)
        self.conv0 = nn.Conv2d(channels_out, channels_mid, kernel_size=1, stride=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels_mid, channels_out, kernel_size=1, stride=1)
        self.sigmoid0 = nn.Sigmoid()

    def forward(self, x):
        x0 = torch.mean(x, dim=(DIM.WIDTH, DIM.HEIGHT), keepdim=True)
        x0 = self.conv0(x0)
        x0 = self.relu0(x0)
        x0 = self.conv1(x0)
        x0 = self.sigmoid0(x0)
        x = torch.mul(x, x0)
        return x


class SpatialZeroPadding(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        if padding > 0:
            raise NotImplementedError("I don't know how to actually pad 0s")
        self.slice = [slice(None) for _ in range(4)]
        self.slice[DIM.HEIGHT] = slice(-padding, padding)
        self.slice[DIM.WIDTH] = slice(-padding, padding)

    def forward(self, x):
        return x[self.slice]

if __name__ == "__main__":
    model_path= r'D:\60-fps-Project\Projects\RIFE GUI\Auxiliary\SuperResolution\waifu2x\waifu2x-cunet2x-gan-real-305k.pth'
    _waifu = UpCunet()
    loadnet = torch.load(model_path, map_location='cpu')
    _waifu.load_state_dict(loadnet, strict=True)
    _waifu.eval()
    device = torch.device('cuda')
    _waifu = _waifu.to(device)

    _image_path = r"D:\60-fps-Project\Projects\RIFE GUI\test\images"
    _img = cv2.imread(os.path.join(_image_path, r'00000000.png'))
    _img = torch.from_numpy(np.transpose(_img, (2, 0, 1))).float()
    _img = _img.unsqueeze(0).to(device)
    _output_img = _waifu(_img)
    _output_img = _output_img.data.squeeze().float().clamp_(0, 1)  # .cpu().numpy()
    _output_img = torch.transpose(_output_img, 0, 2)
    _output_img = torch.transpose(_output_img, 0, 1)
    _output_img = _output_img.mul(255.)
    _output_img = torch.round(_output_img)
    _output = _output_img.cpu().numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(_image_path, f'waifu_cuda.png'), _output)