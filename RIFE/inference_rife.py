import os
import traceback
import warnings

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from Utils.StaticParameters import appDir, RGB_TYPE
from Utils.utils import ArgumentManager, VideoFrameInterpolationBase, Tools

warnings.filterwarnings("ignore")
# from line_profiler_pycharm import profile
"""TEST"""


# os.environ.setdefault('CUDA_LAUNCH_BLOCKING', "1")


class RifeInterpolation(VideoFrameInterpolationBase):
    def __init__(self, __args: ArgumentManager, logger):
        super().__init__(__args, logger)
        self.initiated = False
        self.ARGS = __args

        self.use_auto_scale = self.ARGS.use_rife_auto_scale
        # self.auto_scale_predict_size = self.ARGS.rife_auto_scale_predict_size
        self.device = None
        self.device_count = torch.cuda.device_count()
        self.model = None
        self.model_path = ""
        self.model_version = 0
        self.tta_mode = self.ARGS.rife_tta_mode
        self.tta_iter = self.ARGS.rife_tta_iter

    def initiate_algorithm(self):
        if self.initiated:
            return

        self._initiate_torch()

        self._check_model_path()

        self.logger.info("Loading RIFE Model: https://github.com/hzwer/arXiv2020-RIFE")
        try:
            from RIFE.RIFE_HDv2 import Model
            model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                          forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
            model.load_model(self.model_path, -1 if not self.ARGS.use_rife_multi_cards else 0)
            self.model_version = 2
            self.logger.info("RIFE v2.x model loaded.")
        except:
            try:
                from RIFE.RIFE_HDv3 import Model
                model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                              forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
                model.load_model(self.model_path, -1)
                self.model_version = 3
                self.logger.info("RIFE v3.x model loaded.")
            except:
                from RIFE.RIFE_v6 import Model
                model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                              forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
                model.load_model(self.model_path, -1)
                self.model_version = 6
                self.logger.info("RIFE v6 model loaded.")
        self.model = model
        self.model.eval()
        self.model.device()

        self._print_card_info()
        self.initiated = True

    def _print_card_info(self):
        first_card = torch.cuda.get_device_properties(0)
        card_info = f"{first_card.name}, {first_card.total_memory / 1024 ** 3:.1f} GB"
        self.logger.info(f"RIFE Using {card_info}, model_name: {os.path.basename(self.model_path)}")

    def _check_model_path(self):
        if self.ARGS.rife_model == "" or not os.path.exists(self.ARGS.rife_model):
            self.model_path = os.path.join(appDir, 'train_log', 'official_2.3')
            self.logger.warning("Using Default RIFE Model official_2.3")
        else:
            self.model_path = self.ARGS.rife_model

    def _initiate_torch(self):
        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.ARGS.use_rife_fp16 = False
            self.logger.info("RIFE Using cpu")
        else:
            self.device = torch.device(f"cuda")
            # torch.cuda.set_device(self.ARGS.use_specific_gpu)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            if self.ARGS.use_rife_fp16:
                try:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
                    self.logger.info("RIFE FP16 mode switch success")
                except Exception as e:
                    self.logger.info("RIFE FP16 mode switch failed")
                    traceback.print_exc()
                    self.ARGS.use_rife_fp16 = False

    def __inference(self, img1, img2, scale):
        padding, h, w = self._generate_padding(img1, scale)
        i1 = self._generate_torch_img(img1, padding)
        i2 = self._generate_torch_img(img2, padding)
        if self.ARGS.is_rife_reverse:
            mid = self.model.inference(i2, i1, scale, iter_time=self.tta_iter)
        else:
            mid = self.model.inference(i1, i2, scale, iter_time=self.tta_iter)
        del i1, i2
        mid = ((mid[0] * RGB_TYPE.SIZE).float().cpu().numpy().transpose(1, 2, 0))[:h, :w].copy()
        return mid

    def _make_n_inference(self, img1, img2, scale, n):
        if self.is_interlace_inference:
            pieces_img1 = self.split_input_image(img1)
            pieces_img2 = self.split_input_image(img2)
            pieces_mid = list()
            for piece_img1, piece_img2 in zip(pieces_img1, pieces_img2):
                pieces_mid.append(self.__inference(piece_img1, piece_img2, scale))
            mid = self.sew_input_pieces(pieces_mid, *img1.shape)
        else:
            mid = self.__inference(img1, img2, scale)
        if n == 1:
            return [mid]
        first_half = self._make_n_inference(img1, mid, scale, n=n // 2)
        second_half = self._make_n_inference(mid, img2, scale, n=n // 2)
        if n % 2:
            return [*first_half, mid, *second_half]
        else:
            return [*first_half, *second_half]

    def _generate_padding(self, img, scale: float):
        """

        :param scale:
        :param img: cv2.imread [:, :, ::-1]
        :return:
        """
        h, w, _ = img.shape
        if self.model_version == 4:  # special treat for 4.0 scale issue - Practical-RIFE issue #6
            tmp = max(128, int(128 / scale))
        else:
            tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return padding, h, w

    def _generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """

        """
        Multi Cards Optimization:
        OLS: send several imgs pair according to device_count (2 to be specific)
        HERE: Concat [i1, i2] [i3, i4] and send to rife
        """

        try:
            img_torch = torch.from_numpy(img).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(
                0)
            if self.ARGS.use_rife_fp16:
                img_torch = img_torch.half() / RGB_TYPE.SIZE
            else:
                img_torch = img_torch.float() / RGB_TYPE.SIZE
            if self.ARGS.use_rife_multi_cards and self.device_count > 1:
                if self.device_count % 2 == 0:
                    batch = 2
                else:
                    batch = 3
                img_torch = torch.cat([img_torch for i in range(batch)], dim=0)
            return self._pad_image(img_torch, padding)
        except Exception as e:
            print(img)
            traceback.print_exc()
            raise e

    def _pad_image(self, img, padding):
        # if self.ARGS.use_rife_fp16:
        #     return F.pad(img, padding).half()
        # else:
        return F.pad(img, padding)

    def generate_n_interp(self, img1, img2, n, scale, debug=False):
        if debug:
            output_gen = list()
            for i in range(n):
                output_gen.append(img1)
            return output_gen

        interp_gen = self._make_n_inference(img1, img2, scale, n=n)
        return interp_gen

    def run(self):
        pass


class RifeMultiInterpolation(RifeInterpolation):
    def __init__(self, __args: ArgumentManager, logger):
        super().__init__(__args, logger)

    def initiate_algorithm(self):
        if self.initiated:
            return
        self._initiate_torch()
        self._check_model_path()
        self.logger.info("Loading RIFE Model: https://github.com/hzwer/arXiv2020-RIFE")
        try:
            from RIFE.RIFE_v7_multi import Model
            model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                          forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
            model.load_model(self.model_path, -1 if not self.ARGS.use_rife_multi_cards else 0)
            self.model_version = 7
            self.logger.info("RIFE v7 multi model loaded.")
        except:
            from RIFE.RIFE_HDv4 import Model
            model = Model(use_multi_cards=self.ARGS.use_rife_multi_cards,
                          forward_ensemble=self.ARGS.use_rife_forward_ensemble, tta=self.tta_mode)
            model.load_model(self.model_path, -1)
            self.model_version = 4
            self.logger.info("RIFE 4.x model loaded.")
            pass
        self.model = model
        self.model.eval()
        self.model.device()
        self._print_card_info()
        self.initiated = True

    def _check_model_path(self):
        if self.ARGS.rife_model == "" or not os.path.exists(self.ARGS.rife_model):
            self.model_path = os.path.join(appDir, 'train_log', 'official_4.0')
            self.logger.warning("Using Default RIFE Model official_4.0")
        else:
            self.model_path = self.ARGS.rife_model

    # @profile
    def __inference_n(self, img1, img2, scale, n):
        padding, h, w = self._generate_padding(img1, scale)
        i1 = self._generate_torch_img(img1, padding)
        i2 = self._generate_torch_img(img2, padding)
        if self.ARGS.is_rife_reverse:
            mids = self.model.inference(i2, i1, scale, n)
            mids.reverse()
        else:
            mids = self.model.inference(i1, i2, scale, n)
        del i1, i2
        mids = [((mid[0] * RGB_TYPE.SIZE).float().cpu().numpy().transpose(1, 2, 0))[:h, :w] for mid in mids]
        return mids

    def _make_n_inference(self, img1, img2, scale, n):
        if self.is_interlace_inference:
            pieces_img1 = self.split_input_image(img1)
            pieces_img2 = self.split_input_image(img2)
            pieces_mids = list()
            output_imgs = list()
            for piece_img1, piece_img2 in zip(pieces_img1, pieces_img2):
                pieces_mids.append(self.__inference_n(piece_img1, piece_img2, scale, n))
            for pieces in zip(*pieces_mids):
                output_imgs.append(self.sew_input_pieces(pieces, *img1.shape))
        else:
            output_imgs = self.__inference_n(img1, img2, scale, n)
        return output_imgs


if __name__ == "__main__":
    _image_path = r'D:\60-fps-Project\Projects\RIFE GUI\test\images'
    _i0 = cv2.imread(os.path.join(_image_path, r'0.png'))
    _i1 = cv2.imread(os.path.join(_image_path, r'1.png'))
    _inference_module = RifeMultiInterpolation(
        ArgumentManager({'rife_model': os.path.join(appDir, 'train_log', 'official_4.0')}), Tools.get_logger("", ""))
    _inference_module.initiate_algorithm()
    for i in range(60):
        _imid = _inference_module.generate_n_interp(_i0, _i1, 2, 1.0)
    for _mid_index, _mid in enumerate(_imid):
        cv2.imwrite(os.path.join(_image_path, f'n=2_model=4.0_scale=1_{_mid_index}.png'), _mid)
    pass
