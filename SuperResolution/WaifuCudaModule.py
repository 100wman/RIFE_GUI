import math
import os

import cv2
import numpy as np
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F

from Utils.utils import overtime_reminder_deco, Tools

# from line_profiler_pycharm import profile


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = Tools.get_logger("WaifuCuda", '')


class DIM:
    BATCH = 0
    CHANNEL = 1
    WIDTH = 2
    HEIGHT = 3


@ARCH_REGISTRY.register()
class UpCunet(nn.Module):
    def __init__(self, channels=3):
        super(UpCunet, self).__init__()
        self.cunet_unet1 = CunetUnet1(channels, deconv=True)  # True:2x-16#False:1x-8
        self.cunet_unet2 = CunetUnet2(channels, deconv=False)  # -20
        self.spatial_zero_padding = SpatialZeroPadding(-20)

    def forward(self, x):
        # x = np.pad(x, [(18, 18 ), (18, 18), (0, 0)], mode='reflect')#训练不支持奇数，反正切好了偶数的
        x = F.pad(x, (18, 18, 18, 18), 'reflect')
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


class WaifuCudaer:
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10, half=False):
        # TODO Extract Base Class for refactor
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UpCunet()
        loadnet = torch.load(model_path, map_location='cpu')
        model.load_state_dict(loadnet, strict=True)
        model.eval()
        if self.half:
            self.model = model.half().to(self.device)  # compulsory switch to half mode
        else:
            self.model = model.to(self.device)  # compulsory switch to half mode

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        self.output = self.model(self.img)

    # @profile
    def tile_process(self):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                # print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    # @profile
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().clamp_(0, 1)  # .cpu().numpy()
        output_img = torch.transpose(output_img, 0, 2)
        output_img = torch.transpose(output_img, 0, 1)
        if max_range == 65535:  # 16-bit image
            output_img = torch.tensor(65535.0) * output_img
            output_img = torch.round(output_img)
            output = output_img.cpu().numpy().astype(np.uint16)
            # output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output_img = torch.tensor(255.0) * output_img
            output_img = torch.round(output_img)
            output = output_img.cpu().numpy().astype(np.uint8)
            # output = (output_img * 255.0).round().astype(np.uint8)
        # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        # if img_mode == 'L':
        #     output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        #
        # # ------------------- process the alpha channel if necessary ------------------- #
        # if img_mode == 'RGBA':
        #     if alpha_upsampler == 'realesrgan':
        #         self.pre_process(alpha)
        #         if self.tile_size > 0:
        #             self.tile_process()
        #         else:
        #             self.process()
        #         output_alpha = self.post_process()
        #         output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        #         output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        #         output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        #     else:
        #         h, w = alpha.shape[0:2]
        #         output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)
        #
        #     # merge the alpha channel
        #     output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        #     output_img[:, :, 3] = output_alpha
        #
        # # ------------------------------ return ------------------------------ #
        # if max_range == 65535:  # 16-bit image
        #     output = (output_img * 65535.0).round().astype(np.uint16)
        # else:
        #     output = (output_img * 255.0).round().astype(np.uint8)

        # if outscale is not None and outscale != float(self.scale):
        #     output = cv2.resize(
        #         output, (
        #             int(w_input * outscale),
        #             int(h_input * outscale),
        #         ), interpolation=cv2.INTER_)

        return output, img_mode


class SvfiWaifuCuda:
    def __init__(self, model="", gpu_id=0, precent=90, scale=2, tile=100, resize=(0, 0), half=False):
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.resize_param = resize
        self.scale_exp = 2
        self.scale = scale
        # self.scale = 4
        self.available_scales = [2, 4]
        self.alpha = None
        model_path = os.path.join(app_dir, "ncnn", "sr", "waifuCuda", "models", model)
        self.upscaler = WaifuCudaer(scale=self.scale_exp, model_path=model_path, tile=tile, half=half)
        pass

    # @profile
    @overtime_reminder_deco(100, "WaifuCuda",
                            "Low Super-Resolution speed (>100s per image) detected, Please Consider tweak tilesize or lower output resolution to enhance speed")
    def svfi_process(self, img):
        if self.scale > 1:
            cur_scale = 1
            while cur_scale < self.scale:
                img = self.process(img)
                cur_scale *= self.scale_exp
        return img

    def process(self, img):
        output, img_mode = self.upscaler.enhance(img)
        return output


if __name__ == '__main__':
    test = SvfiWaifuCuda(model="waifu2x-cunet2x-gan-real-305k.pth", )
    # test.svfi_process(cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\Utils\RealESRGAN\input\used\input.png"))
    o = test.svfi_process(
        cv2.imread(r"/test/images/0.png", cv2.IMREAD_UNCHANGED))
    cv2.imwrite("../Utils/out3.png", o)
    # cv2.imshow('test', o)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
