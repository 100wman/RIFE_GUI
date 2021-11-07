import os
from math import ceil

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# from line_profiler_pycharm import profile
from torch.backends import cudnn

from ABME.model import SBMENet, ABMRNet, SynthesisNet
from ABME.utils import warp
from Utils.utils import VideoFrameInterpolationBase, ArgumentManager, appDir

cudnn.benchmark = True
import warnings

warnings.filterwarnings('ignore')


class ABMEInterpolation(VideoFrameInterpolationBase):
    def __init__(self, __args: ArgumentManager):
        super().__init__(__args)

        self.initiated = False
        self.ARGS = __args

        self.auto_scale = self.ARGS.use_rife_auto_scale
        self.device = None
        self.device_count = torch.cuda.device_count()
        self.model = None
        self.model_path = ""
        self.model_version = 0
        self.tta_mode = self.ARGS.rife_tta_mode
        self.divide = 0
        self.model_net = None
        self.model_dir = ""

        self.SBMNet = SBMENet()
        self.ABMNet = ABMRNet()
        self.SynNet = SynthesisNet(False)

    # @profile
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
        self.model_dir = os.path.join(appDir, 'train_log', 'abme_best')

        self.SBMNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'SBME_ckpt.pth'), map_location='cpu'))
        self.ABMNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'ABMR_ckpt.pth'), map_location='cpu'))
        self.SynNet.load_state_dict(torch.load(os.path.join(self.model_dir, 'SynNet_ckpt.pth'), map_location='cpu'))

        for param in self.SBMNet.parameters():
            param.requires_grad = False
        for param in self.ABMNet.parameters():
            param.requires_grad = False
        for param in self.SynNet.parameters():
            param.requires_grad = False

        # if self.ARGS.use_rife_fp16:
        #     self.SBMNet.half()
        #     self.ABMNet.half()
        #     self.SynNet.half()
        self.SBMNet.cuda()
        self.ABMNet.cuda()
        self.SynNet.cuda()
        # self.SBMNet.eval()
        # self.ABMNet.eval()
        # self.SynNet.eval()
        self.initiated = True

    # @profile
    def __inference(self, img1, img2):
        frame1 = img1
        frame3 = img2
        ori_w, ori_h, _ = frame1.shape
        with torch.no_grad():
            # if self.ARGS.use_rife_fp16:
            #     frame1 = TF.to_tensor(frame1).half()
            #     frame3 = TF.to_tensor(frame3).half()
            # else:
            #     frame1 = TF.to_tensor(frame1)
            #     frame3 = TF.to_tensor(frame3)
            # frame1 = torch.from_numpy(img).to(self.device, non_blocking=True).permute(2,0,1).unsqueeze(0)
            # frame3 = torch.from_numpy(img).to(self.device, non_blocking=True).permute(2,0,1).unsqueeze(0)
            frame1 = TF.to_tensor(frame1).type(torch.FloatTensor).unsqueeze(0).cuda()
            frame3 = TF.to_tensor(frame3).type(torch.FloatTensor).unsqueeze(0).cuda()

            H = frame1.shape[2]
            W = frame1.shape[3]

            # 4K video requires GPU memory of more than 24GB. We recommend crop it into 4 regions with some margin.
            # if H < 512:
            #     divisor = 64.
            #     D_factor = 1.
            # else:
            #     divisor = 128.
            #     D_factor = 0.5
            divisor = 64.
            D_factor = 1.

            H_ = int(ceil(H / divisor) * divisor * D_factor)
            W_ = int(ceil(W / divisor) * divisor * D_factor)

            frame1_ = F.interpolate(frame1, (H_, W_), mode='bicubic')
            frame3_ = F.interpolate(frame3, (H_, W_), mode='bicubic')

            SBM = self.SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
            SBM_ = F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

            frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1), return_mask=True)
            frame2_3, Mask2_3 = warp(frame3_, SBM_, return_mask=True)

            frame2_Anchor_ = (frame2_1 + frame2_3) / 2
            frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1 - Mask2_1) + frame2_1 * (1 - Mask2_3))

            Z = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
            Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)

            ABM_bw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM * (-1), Z_.exp())
            ABM_fw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

            SBM_ = F.interpolate(SBM, (H, W), mode='bilinear') * 20.0
            ABM_fw = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
            ABM_bw = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

            SBM_[:, 0, :, :] *= W / float(W_)
            SBM_[:, 1, :, :] *= H / float(H_)
            ABM_fw[:, 0, :, :] *= W / float(W_)
            ABM_fw[:, 1, :, :] *= H / float(H_)
            ABM_bw[:, 0, :, :] *= W / float(W_)
            ABM_bw[:, 1, :, :] *= H / float(H_)

            divisor = 8.
            H_ = int(ceil(H / divisor) * divisor)
            W_ = int(ceil(W / divisor) * divisor)

            Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)

            Syn_inputs = F.interpolate(Syn_inputs, (H_, W_), mode='bilinear')
            Syn_inputs[:, 6, :, :] *= float(W_) / W
            Syn_inputs[:, 7, :, :] *= float(H_) / H
            Syn_inputs[:, 8, :, :] *= float(W_) / W
            Syn_inputs[:, 9, :, :] *= float(H_) / H
            Syn_inputs[:, 10, :, :] *= float(W_) / W
            Syn_inputs[:, 11, :, :] *= float(H_) / H

            result = self.SynNet(Syn_inputs)

            result = F.interpolate(result, (H, W), mode='bicubic')

            mid = result.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            del result
            return mid

    # @profile
    def __make_n_inference(self, img1, img2, scale, n):
        if self.is_interlace_inference:
            pieces_img1 = self.split_input_image(img1)
            pieces_img2 = self.split_input_image(img2)
            pieces_mid = list()
            for piece_img1, piece_img2 in zip(pieces_img1, pieces_img2):
                pieces_mid.append(self.__inference(piece_img1, piece_img2))
            mid = self.sew_input_pieces(pieces_mid, *img1.shape)
        else:
            mid = self.__inference(img1, img2)
        if n == 1:
            return [mid]
        first_half = self.__make_n_inference(img1, mid, scale, n=n // 2)
        second_half = self.__make_n_inference(mid, img2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def generate_n_interp(self, img0, img1, n, scale, debug=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen
        interp_gen = self.__make_n_inference(img0, img1, scale, n=n)
        return interp_gen
        pass


if __name__ == "__main__":
    _abme_arg = ArgumentManager({'rife_interlace_inference': 5})
    _abme_instance = ABMEInterpolation(_abme_arg)
    _abme_instance.initiate_algorithm()
    test_dir = r"D:\60-fps-Project\input or ref\Test\[2]Standard-Hard-Case"
    output_dir = r"D:\60-fps-Project\input or ref\Test\case_output"
    img_paths = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
    img_paths = [img_paths[i:i + 2] for i in range(0, len(img_paths), 2)]
    for i, imgs in enumerate(img_paths):
        # resize = (1280, 720)
        # _img0, _img1 = cv2.resize(cv2.imread(imgs[0]), resize), cv2.resize(cv2.imread(imgs[1]), resize)
        _img0, _img1 = cv2.imread(imgs[0]), cv2.imread(imgs[1])
        _output = _abme_instance.generate_n_interp(_img0, _img1, 1, 4)
        od = os.path.join(output_dir, f"{i:0>2d}")
        os.makedirs(od, exist_ok=True)
        for ii, img in enumerate(_output):
            cv2.imwrite(os.path.join(od, f"{ii:0>8d}.png"), img)
        print(od)
    pass
