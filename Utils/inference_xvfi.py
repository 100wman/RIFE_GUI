import warnings

warnings.filterwarnings("ignore")
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils_xvfi import *
from XVFInet import *
from Utils.utils import ArgumentManager, VideoFrameInterpolationBase
# from line_profiler_pycharm import profile


# Utils = Utils()

class XVFIArgument(ArgumentManager):
    def __init__(self, args: dict):
        super().__init__(args)
        if 'Vimeo' in self.rife_model_name:
            self.S_tst = 1
            self.module_scale_factor = 2
        else:
            self.module_scale_factor = 4
            self.S_tst = 5
        self.img_ch = 3
        self.nf = 64
        self.need_patch = True
        self.patch_size = 64
        self.gpu = self.use_specific_gpu
        self.divide = 2 ** self.S_tst * self.module_scale_factor * 4


class XVFInterpolation(VideoFrameInterpolationBase):
    def __init__(self, __args: XVFIArgument):
        super().__init__(__args)

        self.initiated = False
        self.ARGS = __args

        self.auto_scale = self.ARGS.use_rife_auto_scale
        self.device = None
        self.device_count = torch.cuda.device_count()
        self.model = None
        self.model_path = ""
        self.model_version = 0
        self.tta_mode = self.ARGS.use_rife_tta_mode
        self.divide = 0
        self.model_net = None

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

        self.model_path = os.path.join(self.ARGS.rife_model_dir, self.ARGS.rife_model_name)

        """ Initialize a model """
        checkpoint = torch.load(self.model_path, map_location='cpu')
        print("load model '{}', epoch: {},".format(self.model_path, checkpoint['last_epoch'] + 1))
        self.model_net = XVFInet(self.ARGS).apply(weights_init).to(self.device)  # XVFI.apply...

        # to enable the inbuilt cudnn auto-tuner
        # to find the best algorithm to use for your hardware.
        self.model_net.load_state_dict(checkpoint['state_dict_Model'], strict=True)
        epoch = checkpoint['last_epoch']

        # switch to evaluate mode
        self.model_net.eval()

    # @profile
    def __make_n_inference(self, img1, img2, scale, n):
        with torch.no_grad():
            multiple = n
            t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
            self.ARGS.divide = 2 ** self.ARGS.S_tst * scale * 4
            output_imgs = []
            for testIndex, t_value in enumerate(t):
                t_value = torch.tensor(np.expand_dims(np.array([t_value], dtype=np.float32), 0))
                t_value = Variable(t_value.to(self.device))
                # input_frames = np.array([[img1 / 255., img2 / 255.]])
                input_frames = np.transpose([[img1 / 255., img2 / 255.]], (0, 4, 1, 2, 3))
                input_frames = torch.tensor(input_frames, dtype=torch.float32)  # [1, T, C, H, W]
                # frames = [[[[[][]][[][]]][[[][]][[][]]][[[][]][[][]]]]]
                # input_frames = frames[:, :, :-1, :, :]  # [1,C,T,H,W]
                # if (testIndex % (multiple - 1)) == 0:
                input_frames = Variable(input_frames.to(self.device))
                B, C, T, H, W = input_frames.size()
                H_padding = (self.ARGS.divide - H % self.ARGS.divide) % self.ARGS.divide
                W_padding = (self.ARGS.divide - W % self.ARGS.divide) % self.ARGS.divide
                if H_padding != 0 or W_padding != 0:
                    input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), "constant")

                pred_frameT = self.model_net(input_frames, t_value, is_training=False)
                if H_padding != 0 or W_padding != 0:
                    pred_frameT = pred_frameT[:, :, :H, :W]
                pred_frameT = pred_frameT.data.squeeze().float().clamp_(0, 1)  # .cpu().numpy()
                pred_frameT = torch.transpose(pred_frameT, 0, 2)
                pred_frameT = torch.transpose(pred_frameT, 0, 1)
                pred_frameT = torch.tensor(255.0) * pred_frameT
                pred_frameT = torch.round(pred_frameT)
                output_img = pred_frameT.detach().cpu().numpy().astype(np.uint8)

                # pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())
                # output_img = np.around(np.transpose(pred_frameT, [1, 2, 0]) * 255.)  # [h,w,c] and [-1,1] to [0,255]
                # output_img = np.around(denorm255_np(pred_frameT))  # [h,w,c] and [-1,1] to [0,255]

                output_imgs.append(output_img)
            return output_imgs

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
    _xvfi_arg = XVFIArgument(
        {"rife_model_dir": r"D:\60-fps-Project\Projects\XVFI\checkpoint_dir\XVFInet_X4K1000FPS_exp1",
         "rife_model_name": r"XVFInet_X4K1000FPS_exp1_latest.pt",
         })
    # _xvfi_arg = XVFIArgument(
    #     {"rife_model_dir": r"D:\60-fps-Project\Projects\XVFI\checkpoint_dir\XVFInet_Vimeo_exp1",
    #      "rife_model_name": r"XVFInet_Vimeo_exp1_latest.pt",
    #      })
    _xvfi_instance = XVFInterpolation(_xvfi_arg)
    _xvfi_instance.initiate_algorithm()
    test_dir = r"D:\60-fps-Project\input or ref\Test\[6]XVFI_input"
    output_dir = r"D:\60-fps-Project\input or ref\Test\[6]XVFI_output"
    img_paths = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
    img_paths = [img_paths[i:i + 2] for i in range(0, len(img_paths), 2)]
    for i, imgs in enumerate(img_paths):
        resize = (2000, 1000)
        h, w, c = cv2.imread(imgs[0]).shape
        original_resolution = (w, h)
        # _img0, _img1 = cv2.imread(imgs[0]), cv2.imread(imgs[1])
        _img0, _img1 = cv2.resize(cv2.imread(imgs[0]), resize), cv2.resize(cv2.imread(imgs[1]), resize)
        _output = _xvfi_instance.generate_n_interp(_img0, _img1, 4, 4)
        od = os.path.join(output_dir, f"{i:0>2d}")
        os.makedirs(od, exist_ok=True)
        for ii, img in enumerate(_output):
            cv2.imwrite(os.path.join(od, f"{ii:0>8d}.png"), cv2.resize(img, original_resolution))
    pass
