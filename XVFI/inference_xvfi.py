# from line_profiler_pycharm import profile
import glob
import os
import warnings

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.nn import functional as F

from Utils.utils import ArgumentManager, VideoFrameInterpolationBase
from XVFI.XVFInet import XVFInet
from XVFI.utils import weights_init
from torch.autograd import Variable
# from line_profiler_pycharm import profile


warnings.filterwarnings("ignore")


class XVFIArgument:
    def __init__(self, model_name: str, gpu: int):
        if 'vimeo' in model_name.lower():
            self.S_tst = 1
            self.module_scale_factor = 2
        else:
            """X4K1000FPS"""
            self.S_tst = 5
            self.module_scale_factor = 4
        self.img_ch = 3
        self.nf = 64
        self.need_patch = True
        self.patch_size = 64
        self.gpu = gpu
        self.divide = 2 ** self.S_tst * self.module_scale_factor * 4


class XVFInterpolation(VideoFrameInterpolationBase):
    def __init__(self, __args: ArgumentManager):
        super().__init__(__args)
        self.initiated = False
        self.ARGS = __args
        self.auto_scale = self.ARGS.use_rife_auto_scale
        self.device = None
        self.device_count = torch.cuda.device_count()
        self.tta_mode = self.ARGS.rife_tta_mode

        self.model_path = ""
        self.model_net = None
        self.XVFI_Argument = XVFIArgument(self.ARGS.rife_model_name, self.ARGS.use_specific_gpu)

    def initiate_algorithm(self):
        if self.initiated:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("No Cuda Device available")
        else:
            self.device = torch.device(f"cuda")
            # torch.cuda.set_device(self.ARGS.use_specific_gpu)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        model_path = glob.glob(os.path.join(self.ARGS.rife_model_dir, self.ARGS.rife_model_name, "*.pt"))[0]

        """ Initialize a model """
        checkpoint = torch.load(model_path, map_location='cpu')
        print("INFO - XVFI load model '{}', epoch: {},".format(os.path.basename(model_path),
                                                               checkpoint['last_epoch'] + 1))
        self.model_net = XVFInet(self.XVFI_Argument).apply(weights_init).to(self.device)  # XVFI.apply...

        # to enable the inbuilt cudnn auto-tuner
        # to find the best algorithm to use for your hardware.
        self.model_net.load_state_dict(checkpoint['state_dict_Model'], strict=True)
        # switch to evaluate mode
        self.model_net.eval()

    # @profile
    def __inference(self, img1, img2, n):
        with torch.no_grad():
            multiple = n + 1
            t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
            output_imgs = []
            divide = self.XVFI_Argument.divide
            for testIndex, t_value in enumerate(t):
                t_value = torch.tensor(np.expand_dims(np.array([t_value], dtype=np.float32), 0))
                t_value = Variable(t_value.to(self.device))
                i1 = torch.from_numpy(img1).to(self.device)
                i2 = torch.from_numpy(img2).to(self.device)
                input_frames = torch.stack((i1, i2), dim=0).unsqueeze(0).permute(0, 4, 1, 2, 3) / 255.
                input_frames = Variable(input_frames)
                B, C, T, H, W = input_frames.size()
                H_padding = (divide - H % divide) % divide
                W_padding = (divide - W % divide) % divide
                if H_padding != 0 or W_padding != 0:
                    input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), "constant")

                pred_frameT = self.model_net(input_frames, t_value, is_training=False)
                if H_padding != 0 or W_padding != 0:
                    pred_frameT = pred_frameT[:, :, :H, :W]
                pred_frameT = pred_frameT.data.squeeze().float().clamp_(0, 1).permute(1, 2, 0).mul(
                    255.)  # .cpu().numpy()
                pred_frameT = torch.round(pred_frameT)
                output_img = pred_frameT.detach().cpu().numpy()

                output_imgs.append(output_img)
            return output_imgs

    # @profile
    def __make_n_inference(self, img1, img2, scale, n):
        if self.is_interlace_inference:
            pieces_img1 = self.split_input_image(img1)
            pieces_img2 = self.split_input_image(img2)
            pieces_mids = list()
            output_imgs = list()
            for piece_img1, piece_img2 in zip(pieces_img1, pieces_img2):
                pieces_mids.append(self.__inference(piece_img1, piece_img2, n))
            for pieces in zip(*pieces_mids):
                output_imgs.append(self.sew_input_pieces(pieces, *img1.shape))
        else:
            output_imgs = self.__inference(img1, img2, n)
        return output_imgs

    def generate_n_interp(self, img0, img1, n, scale, debug=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        interp_gen = self.__make_n_inference(img0, img1, scale, n=n)
        return interp_gen


if __name__ == "__main__":
    # _xvfi_arg = ArgumentManager(
    #     {"rife_model_dir": r"D:\60-fps-Project\Projects\XVFI\checkpoint_dir\XVFInet_X4K1000FPS_exp1",
    #      "rife_model_name": r"XVFInet_X4K1000FPS_exp1_latest.pt",
    #      })
    _xvfi_arg = ArgumentManager(
        {"rife_model_dir": r"D:\60-fps-Project\Projects\XVFI\checkpoint_dir",
         "rife_model_name": r"XVFInet_Vimeo_exp1",
         "rife_interlace_inference": 2
         })
    _xvfi_instance = XVFInterpolation(_xvfi_arg)
    _xvfi_instance.initiate_algorithm()
    test_dir = r"D:\60-fps-Project\input or ref\Test\[2]Standard-Hard-Case"
    output_dir = r"D:\60-fps-Project\input or ref\Test\output"
    img_paths = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
    img_paths = [img_paths[i:i + 2] for i in range(0, len(img_paths), 2)]
    for i, imgs in enumerate(img_paths):
        # resize = (2000, 1000)
        h, w, c = cv2.imread(imgs[0]).shape
        original_resolution = (w, h)
        _img0, _img1 = cv2.imread(imgs[0]), cv2.imread(imgs[1])
        # _img0, _img1 = cv2.resize(cv2.imread(imgs[0]), resize), cv2.resize(cv2.imread(imgs[1]), resize)
        _output = _xvfi_instance.generate_n_interp(_img0, _img1, 2, 4)
        od = os.path.join(output_dir, f"{i:0>2d}")
        os.makedirs(od, exist_ok=True)
        for ii, img in enumerate(_output):
            cv2.imwrite(os.path.join(od, f"{ii:0>8d}.png"), cv2.resize(img, original_resolution))
    pass
