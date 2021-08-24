# coding: utf-8
import datetime
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from configparser import ConfigParser, NoOptionError, NoSectionError
from queue import Queue

import cv2
import numpy as np
from sklearn import linear_model
from skvideo.utils import check_output

from steamworks import STEAMWORKS


class AiModulePaths:
    """Relevant to root dir(app dir)"""
    sr_algos = ["ncnn/sr", ]


class SupportFormat:
    img_inputs = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    img_outputs = ['.png', '.tiff', '.jpg']
    vid_outputs = ['.mp4', '.mkv', '.mov']


class EncodePresetAssemply:
    encoder = {
        "CPU": {
            "H264,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "H264,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,8bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "H265,10bit": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "ProRes,422": ["hq", "4444", "4444xq"],
            "ProRes,444": ["hq", "4444", "4444xq"],
        },
        "NVENC":
            {"H264,8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
             "H265,8bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
             "H265,10bit": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"], },
        "NVENCC":
            {"H264,8bit": ["default", "performance", "quality"],
             "H265,8bit": ["default", "performance", "quality"],
             "H265,10bit": ["default", "performance", "quality"], },
        "QSVENCC":
            {"H264,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,8bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"],
             "H265,10bit": ["best", "higher", "high", "balanced", "fast", "faster", "fastest"], },
        "QSV":
            {"H264,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,8bit": ["slow", "fast", "medium", "veryslow", ],
             "H265,10bit": ["slow", "fast", "medium", "veryslow", ], },

    }
    preset = {
        "HEVC": {
            "x265": ["slow", "ultrafast", "fast", "medium", "veryslow"],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
            "QSV": ["slow", "fast", "medium", "veryslow", ],
        },
        "H264": {
            "x264": ["slow", "ultrafast", "fast", "medium", "veryslow", "placebo", ],
            "NVENC": ["slow", "medium", "fast", "hq", "bd", "llhq", "loseless"],
            "QSV": ["slow", "fast", "medium", "veryslow", ],
        },
        "ProRes": ["hq", "4444", "4444xq"]
    }
    pixfmt = {
        "HEVC": {
            "x265": ["yuv420p10le", "yuv420p", "yuv422p", "yuv444p", "yuv422p10le", "yuv444p10le", "yuv420p12le",
                     "yuv422p12le", "yuv444p12le"],
            "NVENC": ["p010le", "yuv420p", "yuv444p", ],
            "QSV": ["yuv420p", "p010le", ],
        },
        "H264": {
            "x264": ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", ],
            "NVENC": ["yuv420p", "yuv444p"],
            "QSV": ["yuv420p", ],
        },
        "ProRes": ["yuv422p10le", "yuv444p10le"]
    }


class SettingsPresets:
    genre = \
        {0:  # 动漫
            {0:  # 速度
                {
                    0: {"encoder": "CPU", "...": "..."},  # CPU
                    1: 1,  # NVENC
                }
            },
            1:  # 实拍
                {0:  # 速度
                     {0: 0,  # CPU
                      1: 1  # NVENC
                      }
                 }
        }
    genre_2 = {(0, 0, 0): {"render_crf": 16}}


class DefaultConfigParser(ConfigParser):
    """
    自定义参数提取
    """

    def get(self, section, option, fallback=None, raw=False):
        try:
            d = self._unify_values(section, None)
        except NoSectionError:
            if fallback is None:
                raise
            else:
                return fallback
        option = self.optionxform(option)
        try:
            value = d[option]
        except KeyError:
            if fallback is None:
                raise NoOptionError(option, section)
            else:
                return fallback

        if type(value) == str and not len(str(value)):
            return fallback

        if type(value) == str and value in ["false", "true"]:
            if value == "false":
                return False
            return True

        return value


class Tools:
    resize_param = (480, 270)
    crop_param = (0, 0, 0, 0)

    def __init__(self):
        self.resize_param = (480, 270)
        self.crop_param = (0, 0, 0, 0)
        pass

    @staticmethod
    def fillQuotation(string):
        if string[0] != '"':
            return f'"{string}"'

    @staticmethod
    def get_logger(name, log_path, debug=False):
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s')
        if ArgumentManager.is_release:
            logger_formatter = logging.Formatter(f'%(asctime)s - %(module)s - %(levelname)s - %(message)s')

        log_path = os.path.join(log_path, "log")  # private dir for logs
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger_path = os.path.join(log_path,
                                   f"{datetime.datetime.now().date()}.log")

        txt_handler = logging.FileHandler(logger_path, encoding='utf-8')
        txt_handler.setFormatter(logger_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logger_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(txt_handler)
        return logger

    @staticmethod
    def make_dirs(dir_lists, rm=False):
        for d in dir_lists:
            if rm and os.path.exists(d):
                shutil.rmtree(d)
                continue
            if not os.path.exists(d):
                os.mkdir(d)
        pass

    @staticmethod
    def gen_next(gen: iter):
        try:
            return next(gen)
        except StopIteration:
            return None

    @staticmethod
    def clean_parsed_config(args: dict) -> dict:
        for a in args:
            if args[a] in ["false", "true"]:
                if args[a] == "false":
                    args[a] = False
                else:
                    args[a] = True
                continue
            try:
                tmp = float(args[a])
                try:
                    if not tmp - int(args[a]):
                        tmp = int(args[a])
                except ValueError:
                    pass
                args[a] = tmp
                continue
            except ValueError:
                pass
            if not len(args[a]):
                print(f"Warning: Find Empty Args at '{a}'")
                args[a] = ""
        return args
        pass

    @staticmethod
    def check_pure_img(img1):
        if np.var(img1) < 10:
            return True
        return False

    @staticmethod
    def get_norm_img(img1, resize=True):
        if resize:
            img1 = cv2.resize(img1, Tools.resize_param, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.equalizeHist(img1)  # 进行直方图均衡化
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img1

    @staticmethod
    def get_norm_img_diff(img1, img2, resize=True) -> float:
        """
        Normalize Difference
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return: float
        """
        img1 = Tools.get_norm_img(img1, resize)
        img2 = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        diff = cv2.absdiff(img1, img2).mean()
        return diff

    @staticmethod
    def get_norm_img_flow(img1, img2, resize=True, flow_thres=1) -> (int, np.array):
        """
        Normalize Difference
        :param flow_thres: 光流移动像素长
        :param resize:
        :param img1: cv2
        :param img2: cv2
        :return:  (int, np.array)
        """
        prevgray = Tools.get_norm_img(img1, resize)
        gray = Tools.get_norm_img(img2, resize)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 使用Gunnar Farneback算法计算密集光流
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 绘制线
        step = 10
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []
        flow_cnt = 0

        for l in lines:
            if math.sqrt(math.pow(l[0][0] - l[1][0], 2) + math.pow(l[0][1] - l[1][1], 2)) > flow_thres:
                flow_cnt += 1
                line.append(l)

        cv2.polylines(prevgray, line, 0, (0, 255, 255))
        comp_stack = np.hstack((prevgray, gray))
        return flow_cnt, comp_stack

    @staticmethod
    def get_filename(path):
        if not os.path.isfile(path):
            return os.path.basename(path)
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def get_exp_edge(num):
        b = 2
        scale = 0
        while num > b ** scale:
            scale += 1
        return scale

    @staticmethod
    def get_mixed_scenes(img0, img1, n):
        """
        return n-1 images
        :param img0:
        :param img1:
        :param n:
        :return:
        """
        step = 1 / n
        beta = 0
        output = list()
        for _ in range(n - 1):
            beta += step
            alpha = 1 - beta
            mix = cv2.addWeighted(img0[:, :, ::-1], alpha, img1[:, :, ::-1], beta, 0)[:, :, ::-1].copy()
            output.append(mix)
        return output

    @staticmethod
    def get_fps(path: str):
        """
        Get Fps from path
        :param path:
        :return: fps float
        """
        if not os.path.isfile(path):
            return 0
        try:
            if not os.path.isfile(path):
                input_fps = 0
            else:
                input_stream = cv2.VideoCapture(path)
                input_fps = input_stream.get(cv2.CAP_PROP_FPS)
            return input_fps
        except Exception:
            return 0

    @staticmethod
    def get_existed_chunks(project_dir: str):
        chunk_paths = []
        for chunk_p in os.listdir(project_dir):
            if re.match("chunk-\d+-\d+-\d+\.\w+", chunk_p):
                chunk_paths.append(chunk_p)

        if not len(chunk_paths):
            return chunk_paths, -1, -1

        chunk_paths.sort()
        last_chunk = chunk_paths[-1]
        chunk_cnt, last_frame = re.findall('chunk-(\d+)-\d+-(\d+).*?', last_chunk)[0]
        return chunk_paths, int(chunk_cnt), int(last_frame)

    @staticmethod
    def popen(args: str):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        p = subprocess.Popen(args, startupinfo=startupinfo)
        return p

    @staticmethod
    def md5(d: str):
        m = hashlib.md5(d.encode(encoding='utf-8'))
        return m.hexdigest()


class ImgSeqIO:
    def __init__(self, folder=None, is_read=True, thread=4, is_tool=False, start_frame=0, logger=None,
                 output_ext=".png", exp=2, resize=(0, 0), is_esr=False, **kwargs):
        if logger is None:
            self.logger = Tools.get_logger(name="ImgIO", log_path=folder)
        else:
            self.logger = logger

        if output_ext[0] != ".":
            output_ext = "." + output_ext
        self.output_ext = output_ext

        if folder is None or os.path.isfile(folder):
            self.logger.error(f"Invalid ImgSeq Folder: {folder}")
            return
        self.seq_folder = folder  # + "/tmp"  # weird situation, cannot write to target dir, father dir instead
        if not os.path.exists(self.seq_folder):
            os.makedirs(self.seq_folder, exist_ok=True)
            start_frame = 0
        elif start_frame == -1 and not is_read:
            start_frame = self.get_start_frame()
            # write: start writing at the end of sequence

        self.start_frame = start_frame
        self.frame_cnt = 0
        self.img_list = list()

        self.write_queue = Queue(maxsize=1000)
        self.thread_cnt = thread
        self.thread_pool = list()

        self.use_imdecode = False
        self.resize = resize
        self.resize_flag = all(self.resize)
        self.is_esr = is_esr

        self.exp = exp

        if is_tool:
            return
        if is_read:
            img_list = os.listdir(self.seq_folder)
            img_list.sort()
            for p in img_list:
                fn, ext = os.path.splitext(p)
                if ext.lower() in SupportFormat.img_inputs:
                    if self.frame_cnt < start_frame:
                        self.frame_cnt += 1  # update frame_cnt
                        continue  # do not read frame until reach start_frame img
                    self.img_list.append(os.path.join(self.seq_folder, p))
            self.logger.debug(f"Load {len(self.img_list)} frames at {self.frame_cnt}")
        else:
            """Write Img"""
            self.frame_cnt = start_frame
            self.logger.debug(f"Start Writing {self.output_ext} at No. {self.frame_cnt}")
            for t in range(self.thread_cnt):
                _t = threading.Thread(target=self.write_buffer, name=f"[IMG.IO] Write Buffer No.{t + 1}")
                self.thread_pool.append(_t)
            for _t in self.thread_pool:
                _t.start()

    def get_start_frame(self):
        """
        Get Start Frame when start_frame is at its default value
        :return:
        """
        img_list = list()
        for f in os.listdir(self.seq_folder):
            fn, ext = os.path.splitext(f)
            if ext in SupportFormat.img_inputs:
                img_list.append(int(fn))
        if not len(img_list):
            return 0
        img_list.sort()
        last_img = img_list[-1]  # biggest
        return last_img

    def get_frames_cnt(self):
        """
        Get Frames Cnt with EXP
        :return:
        """
        return len(self.img_list) * 2 ** self.exp

    def read_frame(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_LANCZOS4)
        return img

    def write_frame(self, img, path):
        if self.resize_flag:
            img = cv2.resize(img, (self.resize[0], self.resize[1]))
        cv2.imencode(self.output_ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(path)
        # good

    def nextFrame(self):
        for p in self.img_list:
            img = self.read_frame(p)
            for e in range(2 ** self.exp):
                yield img

    def write_buffer(self):
        while True:
            img_data = self.write_queue.get()
            if img_data[1] is None:
                self.logger.debug(f"{threading.current_thread().name}: get None, break")
                break
            self.write_frame(img_data[1], img_data[0])

    def writeFrame(self, img):
        img_path = os.path.join(self.seq_folder, f"{self.frame_cnt:0>8d}{self.output_ext}")
        img_path = img_path.replace("\\", "/")
        if img is None:
            for t in range(self.thread_cnt):
                self.write_queue.put((img_path, None))
            return
        self.write_queue.put((img_path, img))
        self.frame_cnt += 1
        return

    def close(self):
        for t in range(self.thread_cnt):
            self.write_queue.put(("", None))
        for _t in self.thread_pool:
            while _t.is_alive():
                time.sleep(0.2)
        # if os.path.exists(self.seq_folder):
        #     shutil.rmtree(self.seq_folder)
        return


class RifeInterpolation:
    """Rife 补帧 抽象类"""

    def __init__(self, __args):
        self.initiated = False
        self.args = {}
        if __args is not None:
            """Update Args"""
            self.args = __args
        else:
            raise NotImplementedError("Args not sent in")

        self.device = None
        self.model = None
        self.model_path = ""
        pass

    def initiate_rife(self, __args=None):
        raise NotImplementedError("Abstract")

    def __make_inference(self, img1, img2, scale, exp):
        raise NotImplementedError("Abstract")

    def __make_n_inference(self, img1, img2, scale, n):
        raise NotImplementedError("Abstract")

    def generate_padding(self, img, scale):
        raise NotImplementedError("Abstract")

    def generate_torch_img(self, img, padding):
        """
        :param img: cv2.imread [:, :, ::-1]
        :param padding:
        :return:
        """
        raise NotImplementedError("Abstract")

    def pad_image(self, img, padding):
        raise NotImplementedError("Abstract")

    def generate_interp(self, img1, img2, exp, scale, n=None, debug=False, test=False):
        """

        :param img1: cv2.imread
        :param img2:
        :param exp:
        :param scale:
        :param n:
        :param debug:
        :return: list of interp cv2 image
        """
        raise NotImplementedError("Abstract")

    def generate_n_interp(self, img1, img2, n, scale, debug=False):
        raise NotImplementedError("Abstract")

    def get_auto_scale(self, img1, img2, scale):
        raise NotImplementedError("Abstract")

    def run(self):
        raise NotImplementedError("Abstract")


class SuperResolution:
    """
    超分抽象类
    """

    def __init__(
            self,
            gpuid=0,
            model="models-cunet",
            tta_mode=False,
            num_threads=1,
            scale: float = 2,
            noise=0,
            tilesize=0,
    ):
        self.tilesize = tilesize
        self.noise = noise
        self.scale = scale
        self.num_threads = num_threads
        self.tta_mode = tta_mode
        self.model = model
        self.gpuid = gpuid

    def process(self, im):
        return im

    def svfi_process(self, img):
        """
        SVFI 用于超分的接口
        :param img:
        :return:
        """
        return img


class PathManager:
    """
    路径管理器
    """

    def __init__(self):
        pass


class ArgumentManager:
    """
    For OLS's arguments input management
    """
    app_id = 1692080
    pro_dlc_id = 1718750

    """Release Version Control"""
    is_steam = False
    is_free = False
    is_release = True
    traceback_limit = 0 if is_release else None
    gui_version = "3.5.8"
    version_tag = f"{gui_version} " \
                  f"{'Professional' if not is_free else 'Community'} [{'Steam' if is_steam else 'No Steam'}]"
    ols_version = "6.9.10"
    """ 发布前改动以上参数即可 """

    path_len_limit = 230

    def __init__(self, args: dict):
        self.app_dir = args.get("app_dir", "")
        self.ols_path = args.get("ols_path", "")
        self.batch = args.get("batch", False)
        self.ffmpeg = args.get("ffmpeg", "")

        self.config = args.get("config", "")
        self.input = args.get("input", "")
        self.output_dir = args.get("output_dir", "")
        self.task_id = args.get("task_id", "")
        self.gui_inputs = args.get("gui_inputs", "")
        self.input_fps = args.get("input_fps", 0)
        self.target_fps = args.get("target_fps", 0)
        self.output_ext = args.get("output_ext", ".mp4")
        self.is_img_input = args.get("is_img_input", False)
        self.is_img_output = args.get("is_img_output", False)
        self.is_output_only = args.get("is_output_only", True)
        self.is_save_audio = args.get("is_save_audio", True)
        self.input_start_point = args.get("input_start_point", None)
        self.input_end_point = args.get("input_end_point", None)
        if self.input_start_point == "00:00:00":
            self.input_start_point = None
        if self.input_end_point == "00:00:00":
            self.input_end_point = None
        self.output_chunk_cnt = args.get("output_chunk_cnt", 0)
        self.interp_start = args.get("interp_start", 0)
        self.risk_resume_mode = args.get("risk_resume_mode", True)

        self.is_no_scdet = args.get("is_no_scdet", False)
        self.is_scdet_mix = args.get("is_scdet_mix", False)
        self.use_scdet_fixed = args.get("use_scdet_fixed", False)
        self.is_scdet_output = args.get("is_scdet_output", True)
        self.scdet_threshold = args.get("scdet_threshold", 10)
        self.scdet_fixed_max = args.get("scdet_fixed_max", 40)
        self.scdet_flow_cnt = args.get("scdet_flow_cnt", 4)
        self.scdet_mode = args.get("scdet_mode", 0)
        self.remove_dup_mode = args.get("remove_dup_mode", 0)
        self.remove_dup_threshold = args.get("remove_dup_threshold", 0.1)

        self.use_manual_buffer = args.get("use_manual_buffer", False)
        self.manual_buffer_size = args.get("manual_buffer_size", 1)

        self.resize_width = args.get("resize_width", "")
        self.resize_height = args.get("resize_height", "")
        self.resize = args.get("resize", "")
        self.resize_exp = args.get("resize_exp", 1)
        self.crop_width = args.get("crop_width", "")
        self.crop_height = args.get("crop_height", "")
        self.crop = args.get("crop", "")

        self.use_sr = args.get("use_sr", False)
        self.use_sr_algo = args.get("use_sr_algo", "")
        self.use_sr_model = args.get("use_sr_model", "")
        self.use_sr_mode = args.get("use_sr_mode", "")
        self.sr_tilesize = args.get("sr_tilesize", 200)

        self.render_gap = args.get("render_gap", 1000)
        self.use_crf = args.get("use_crf", True)
        self.use_bitrate = args.get("use_bitrate", False)
        self.render_crf = args.get("render_crf", 14)
        self.render_bitrate = args.get("render_bitrate", 90)
        self.render_encoder_preset = args.get("render_encoder_preset", "slow")
        self.render_encoder = args.get("render_encoder", "")
        self.render_hwaccel_mode = args.get("render_hwaccel_mode", "")
        self.render_hwaccel_preset = args.get("render_hwaccel_preset", "")
        self.use_hwaccel_decode = args.get("use_hwaccel_decode", True)
        self.use_manual_encode_thread = args.get("use_manual_encode_thread", False)
        self.render_encode_thread = args.get("render_encode_thread", 16)
        self.is_quick_extract = args.get("is_quick_extract", True)
        self.hdr_mode = args.get("hdr_mode", 0)
        self.render_ffmpeg_customized = args.get("render_ffmpeg_customized", "")
        self.is_no_concat = args.get("is_no_concat", False)
        self.use_fast_denoise = args.get("use_fast_denoise", False)
        self.gif_loop = args.get("gif_loop", True)
        self.is_render_slow_motion = args.get("is_render_slow_motion", False)
        self.render_slow_motion_fps = args.get("render_slow_motion_fps", 0)
        self.use_deinterlace = args.get("use_deinterlace", False)

        self.use_ncnn = args.get("use_ncnn", False)
        self.ncnn_thread = args.get("ncnn_thread", 4)
        self.ncnn_gpu = args.get("ncnn_gpu", 0)
        self.use_rife_tta_mode = args.get("use_rife_tta_mode", False)
        self.use_rife_fp16 = args.get("use_rife_fp16", False)
        self.rife_scale = args.get("rife_scale", 1.0)
        self.rife_model_dir = args.get("rife_model_dir", "")
        self.rife_model = args.get("rife_model", "")
        self.rife_model_name = args.get("rife_model_name", "")
        self.rife_exp = args.get("rife_exp", 1.0)
        self.rife_cuda_cnt = args.get("rife_cuda_cnt", 0)
        self.is_rife_reverse = args.get("is_rife_reverse", False)
        self.use_specific_gpu = args.get("use_specific_gpu", 0)  # !
        self.use_rife_auto_scale = args.get("use_rife_auto_scale", False)
        self.use_rife_forward_ensemble = args.get("use_rife_forward_ensemble", False)
        self.use_rife_multi_cards = args.get("use_rife_multi_cards", False)

        self.debug = args.get("debug", False)
        self.multi_task_rest = args.get("multi_task_rest", False)
        self.multi_task_rest_interval = args.get("multi_task_rest_interval", 1)
        self.after_mission = args.get("after_mission", False)
        self.force_cpu = args.get("force_cpu", False)
        self.expert_mode = args.get("expert_mode", False)
        self.preview_args = args.get("preview_args", False)
        self.pos = args.get("pos", "")
        self.size = args.get("size", "")

        """OLS Params"""
        self.concat_only = args.get("concat_only", False)
        self.extract_only = args.get("extract_only", False)
        self.render_only = args.get("render_only", False)
        self.version = args.get("version", "0.0.0 beta")


class VideoInfo:
    def __init__(self, file_input: str, logger: Tools.get_logger, project_dir: str, ffmpeg=None, img_input=False,
                 hdr_mode=False, exp=0, **kwargs):
        self.filepath = file_input
        self.img_input = img_input
        self.hdr_mode = -1
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
        self.hdr10_parser = "hdr10plus_parser"
        self.logger = logger
        self.project_dir = project_dir
        if ffmpeg is not None:
            self.ffmpeg = Tools.fillQuotation(os.path.join(ffmpeg, "ffmpeg.exe"))
            self.ffprobe = Tools.fillQuotation(os.path.join(ffmpeg, "ffprobe.exe"))
            self.hdr10_parser = Tools.fillQuotation(os.path.join(ffmpeg, "hdr10plus_parser.exe"))
        if not os.path.exists(self.ffmpeg):
            self.ffmpeg = "ffmpeg"
            self.ffprobe = "ffprobe"
            self.hdr10_parser = "hdr10plus_parser"
        self.color_info = dict()
        self.exp = exp
        self.frames_cnt = 0
        self.frames_size = (0, 0)  # width, height
        self.fps = 0
        self.duration = 0
        self.video_info = dict()
        self.update_info()

    def update_hdr_mode(self):
        # TODO Check DV
        if any([i in str(self.video_info["video_info"]) for i in ['dv_profile', 'DOVI']]):
            # Dolby Vision
            self.hdr_mode = 3
            return
        if "color_transfer" not in self.video_info["video_info"]:
            self.logger.warning("Not Find Color Transfer")
            self.hdr_mode = 0
            return
        color_trc = self.video_info["video_info"]["color_transfer"]
        if "smpte2084" in color_trc or "bt2020" in color_trc:
            """should be 10bit encoding"""
            self.hdr_mode = 1  # hdr(normal)
            self.logger.warning("HDR Content Detected")
            if any([i in str(self.video_info["video_info"]).lower()]
                   for i in ['mastering-display', "mastering display", "content light level metadata"]):
                self.hdr_mode = 2  # hdr10
                self.logger.warning("HDR10+ Content Detected")
                check_command = (f'{self.ffmpeg} -loglevel panic -i {Tools.fillQuotation(self.filepath)} -c:v copy '
                                 f'-vbsf hevc_mp4toannexb -f hevc - | '
                                 f'{self.hdr10_parser} -o {os.path.join(self.project_dir, "hdr10plus_metadata.json")} -')
                try:
                    check_output(shlex.split(check_command), shell=True)
                except Exception:
                    self.logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))

        elif "arib-std-b67" in color_trc:
            self.hdr_mode = 4  # HLG
            self.logger.warning("HLG Content Detected")
        pass

    def update_frames_info_ffprobe(self):
        check_command = (f'{self.ffprobe} -v error -show_streams -select_streams v:0 -v error '
                         f'-show_entries stream=index,width,height,r_frame_rate,nb_frames,duration,'
                         f'color_primaries,color_range,color_space,color_transfer -print_format json '
                         f'{Tools.fillQuotation(self.filepath)}')
        result = check_output(shlex.split(check_command))
        try:
            video_info = json.loads(result)["streams"][0]  # select first video stream as input
        except Exception as e:
            self.logger.warning(f"Parse Video Info Failed: {result}")
            raise e
        self.video_info = video_info
        self.video_info['video_info'] = video_info
        self.logger.info(f"\nInput Video Info\n{video_info}")
        # update color info
        if "color_range" in video_info:
            self.color_info["-color_range"] = video_info["color_range"]
        if "color_space" in video_info:
            self.color_info["-colorspace"] = video_info["color_space"]
        if "color_transfer" in video_info:
            self.color_info["-color_trc"] = video_info["color_transfer"]
        if "color_primaries" in video_info:
            self.color_info["-color_primaries"] = video_info["color_primaries"]

        self.update_hdr_mode()

        # update frame size info
        if 'width' in video_info and 'height' in video_info:
            self.frames_size = (int(video_info['width']), int(video_info['height']))

        if "r_frame_rate" in video_info:
            fps_info = video_info["r_frame_rate"].split('/')
            self.fps = int(fps_info[0]) / int(fps_info[1])
            self.logger.info(f"Auto Find FPS in r_frame_rate: {self.fps}")
        else:
            self.logger.warning("Auto Find FPS Failed")
            return False

        if "nb_frames" in video_info:
            self.frames_cnt = int(video_info["nb_frames"])
            self.logger.info(f"Auto Find frames cnt in nb_frames: {self.frames_cnt}")
        elif "duration" in video_info:
            self.duration = float(video_info["duration"])
            self.frames_cnt = round(float(self.duration * self.fps))
            self.logger.info(f"Auto Find Frames Cnt by duration deduction: {self.frames_cnt}")
        else:
            self.logger.warning("FFprobe Not Find Frames Cnt")
            return False
        return True

    def update_frames_info_cv2(self):
        # if not os.path.isfile(self.filepath):
        #     height, width = 0, 0
        video_input = cv2.VideoCapture(self.filepath)
        if not self.fps:
            self.fps = video_input.get(cv2.CAP_PROP_FPS)
        if not self.frames_cnt:
            self.frames_cnt = video_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.duration:
            self.duration = self.frames_cnt / self.fps
        self.frames_size = (
            round(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), round(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def update_info(self):
        if self.img_input:
            if os.path.isfile(self.filepath):
                self.filepath = os.path.dirname(self.filepath)
            seqlist = os.listdir(self.filepath)
            self.frames_cnt = len(seqlist) * 2 ** self.exp
            img = cv2.imdecode(np.fromfile(os.path.join(self.filepath, seqlist[0]), dtype=np.uint8), 1)[:, :,
                  ::-1].copy()
            self.frames_size = (int(img.shape[1]), int(img.shape[0]))
            return
        self.update_frames_info_ffprobe()
        self.update_frames_info_cv2()

    def get_info(self):
        get_dict = {}
        get_dict.update(self.color_info)
        get_dict.update({"video_info": self.video_info})
        get_dict["fps"] = self.fps
        get_dict["size"] = self.frames_size
        get_dict["cnt"] = self.frames_cnt
        get_dict["duration"] = self.duration
        get_dict['hdr_mode'] = self.hdr_mode
        return get_dict


class TransitionDetection_ST:
    def __init__(self, project_dir, scene_queue_length, scdet_threshold=50, no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, scdet_output=False):
        """
        转场检测类
        :param scdet_flow: 输入光流模式：0：2D 1：3D
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        """
        self.scdet_output = scdet_output
        self.scdet_threshold = scdet_threshold
        self.use_fixed_scdet = use_fixed_scdet
        if self.use_fixed_scdet:
            self.scdet_threshold = fixed_max_scdet
        self.scdet_cnt = 0
        self.scene_stack_len = scene_queue_length
        self.absdiff_queue = deque(maxlen=self.scene_stack_len)  # absdiff队列
        self.black_scene_queue = deque(maxlen=self.scene_stack_len)  # 黑场开场特判队列
        self.scene_checked_queue = deque(maxlen=self.scene_stack_len // 2)  # 已判断的转场absdiff特判队列
        self.utils = Tools
        self.dead_thres = 80
        self.born_thres = 2
        self.img1 = None
        self.img2 = None
        self.scdet_cnt = 0
        self.scene_dir = os.path.join(project_dir, "scene")
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)
        self.scene_stack = Queue(maxsize=scene_queue_length)
        self.no_scdet = no_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.absdiff_queue))).reshape(-1, 1), np.array(self.absdiff_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.absdiff_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.absdiff_queue)
        sub_array = diff_array - coef_array
        return sub_array.var() ** 0.65

    def __judge_mean(self, diff):
        var_before = self.__check_var()
        self.absdiff_queue.append(diff)
        var_after = self.__check_var()
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres:
            """Detect new scene"""
            self.scdet_cnt += 1
            self.save_scene(
                f"diff: {diff:.3f}, var_a: {var_before:.3f}, var_b: {var_after:.3f}, cnt: {self.scdet_cnt}")
            self.absdiff_queue.clear()
            self.scene_checked_queue.append(diff)
            return True
        else:
            if diff > self.dead_thres:
                self.absdiff_queue.clear()
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
                self.scene_checked_queue.append(diff)
                return True
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.save_scene(title)

    def save_scene(self, title):
        if not self.scdet_output:
            return
        try:
            comp_stack = np.hstack((self.img1, self.img2))
            comp_stack = cv2.resize(comp_stack, (960, int(960 * comp_stack.shape[0] / comp_stack.shape[1])), )
            cv2.putText(comp_stack,
                        title,
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            if "pure" in title.lower():
                path = f"{self.scdet_cnt:08d}_pure.png"
            elif "band" in title.lower():
                path = f"{self.scdet_cnt:08d}_band.png"
            else:
                path = f"{self.scdet_cnt:08d}.png"
            path = os.path.join(self.scene_dir, path)
            if os.path.exists(path):
                os.remove(path)
            cv2.imencode('.png', cv2.cvtColor(comp_stack, cv2.COLOR_RGB2BGR))[1].tofile(path)
            return
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
            cv2.moveWindow(title, 500, 500)
            cv2.resizeWindow(title, 1920, 540)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    def check_scene(self, _img1, _img2, add_diff=False, no_diff=False, use_diff=-1, **kwargs) -> bool:
        """
        Check if current scene is scene
        :param use_diff:
        :param _img2:
        :param _img1:
        :param add_diff:
        :param no_diff: check after "add_diff" mode
        :return: 是转场则返回真
        """
        img1 = _img1.copy()
        img2 = _img2.copy()
        self.img1 = img1
        self.img2 = img2

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.scdet_threshold:
                return False
            else:
                self.scdet_cnt += 1
                self.save_scene(f"diff: {diff:.3f}, Fix Scdet, cnt: {self.scdet_cnt}")
                return True

        """检测开头黑场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Pure Scene, cnt: {self.scdet_cnt}")
            # self.save_flow()
            return True

        # Check really hard scene at the beginning
        if diff > self.dead_thres:
            self.absdiff_queue.clear()
            self.scdet_cnt += 1
            self.save_scene(f"diff: {diff:.3f}, Dead Scene, cnt: {self.scdet_cnt}")
            self.scene_checked_queue.append(diff)
            return True

        if len(self.absdiff_queue) < self.scene_stack_len or add_diff:
            if diff not in self.absdiff_queue:
                self.absdiff_queue.append(diff)
            return False

        """Duplicate Frames Special Judge"""
        if no_diff and len(self.absdiff_queue):
            self.absdiff_queue.pop()
            if not len(self.absdiff_queue):
                return False

        """Judge"""
        return self.__judge_mean(diff)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class TransitionDetection:
    def __init__(self, scene_queue_length, scdet_threshold=50, project_dir="", no_scdet=False,
                 use_fixed_scdet=False, fixed_max_scdet=50, remove_dup_mode=0, scdet_output=False, scdet_flow=0,
                 **kwargs):
        """
        转场检测类
        :param scdet_flow: 输入光流模式：0：2D 1：3D
        :param scene_queue_length: 转场判定队列长度
        :param fixed_scdet:
        :param scdet_threshold: （标准输入）转场阈值
        :param output: 输出
        :param no_scdet: 不进行转场识别
        :param use_fixed_scdet: 使用固定转场阈值
        :param fixed_max_scdet: 使用的最大转场阈值
        :param kwargs:
        """
        self.view = False
        self.utils = Tools
        self.scdet_cnt = 0
        self.scdet_threshold = scdet_threshold
        self.scene_dir = os.path.join(project_dir, "scene")  # 存储转场图片的文件夹路径
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)

        self.dead_thres = 80  # 写死最高的absdiff
        self.born_thres = 3  # 写死判定为非转场的最低阈值

        self.scene_queue_len = scene_queue_length
        if remove_dup_mode in [1, 2]:
            """去除重复帧一拍二或N"""
            self.scene_queue_len = 8  # 写死

        self.flow_queue = deque(maxlen=self.scene_queue_len)  # flow_cnt队列
        self.black_scene_queue = deque(maxlen=self.scene_queue_len)  # 黑场景特判队列
        self.absdiff_queue = deque(maxlen=self.scene_queue_len)  # absdiff队列
        self.scene_stack = Queue(maxsize=self.scene_queue_len)  # 转场识别队列

        self.no_scdet = no_scdet
        self.use_fixed_scdet = use_fixed_scdet
        self.scedet_info = {"scene": 0, "normal": 0, "dup": 0, "recent_scene": -1}
        # 帧种类，scene为转场，normal为正常帧，dup为重复帧，即两帧之间的计数关系

        self.img1 = None
        self.img2 = None
        self.flow_img = None
        self.before_img = None
        if self.use_fixed_scdet:
            self.dead_thres = fixed_max_scdet

        self.scene_output = scdet_output
        if scdet_flow == 0:
            self.scdet_flow = 3
        else:
            self.scdet_flow = 1

        self.now_absdiff = -1
        self.now_vardiff = -1
        self.now_flow_cnt = -1

    def __check_coef(self):
        reg = linear_model.LinearRegression()
        reg.fit(np.array(range(len(self.flow_queue))).reshape(-1, 1), np.array(self.flow_queue).reshape(-1, 1))
        return reg.coef_, reg.intercept_

    def __check_var(self):
        """
        计算“转场”方差
        :return:
        """
        coef, intercept = self.__check_coef()
        coef_array = coef * np.array(range(len(self.flow_queue))).reshape(-1, 1) + intercept
        diff_array = np.array(self.flow_queue)
        sub_array = np.abs(diff_array - coef_array)
        return sub_array.var() ** 0.65

    def __judge_mean(self, flow_cnt, diff, flow):
        # absdiff_mean = 0
        # if len(self.absdiff_queue) > 1:
        #     self.absdiff_queue.pop()
        #     absdiff_mean = np.mean(self.absdiff_queue)

        var_before = self.__check_var()
        self.flow_queue.append(flow_cnt)
        var_after = self.__check_var()
        self.now_absdiff = diff
        self.now_vardiff = var_after - var_before
        self.now_flow_cnt = flow_cnt
        if var_after - var_before > self.scdet_threshold and diff > self.born_thres and flow_cnt > np.mean(
                self.flow_queue):
            """Detect new scene"""
            self.see_flow(
                f"flow_cnt: {flow_cnt:.3f}, diff: {diff:.3f}, before: {var_before:.3f}, after: {var_after:.3f}, "
                f"cnt: {self.scdet_cnt + 1}", flow)
            self.flow_queue.clear()
            self.scdet_cnt += 1
            self.save_flow()
            return True
        else:
            if diff > self.dead_thres:
                """不漏掉死差转场"""
                self.flow_queue.clear()
                self.see_result(f"diff: {diff:.3f}, False Alarm, cnt: {self.scdet_cnt + 1}")
                self.scdet_cnt += 1
                self.save_flow()
                return True
            # see_result(f"compare: False, diff: {diff}, bm: {before_measure}")
            self.absdiff_queue.append(diff)
            return False

    def end_view(self):
        self.scene_stack.put(None)
        while True:
            scene_data = self.scene_stack.get()
            if scene_data is None:
                return
            title = scene_data[0]
            scene = scene_data[1]
            self.see_result(title)

    def see_result(self, title):
        """捕捉转场帧预览"""
        if not self.view:
            return
        comp_stack = np.hstack((self.img1, self.img2))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_flow(self):
        if not self.scene_output:
            return
        try:
            cv2.putText(self.flow_img,
                        f"diff: {self.now_absdiff:.2f}, vardiff: {self.now_vardiff:.2f}, flow: {self.now_flow_cnt:.2f}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
            cv2.imencode('.png', cv2.cvtColor(self.flow_img, cv2.COLOR_RGB2BGR))[1].tofile(
                os.path.join(self.scene_dir, f"{self.scdet_cnt:08d}.png"))
        except Exception:
            traceback.print_exc()
        pass

    def see_flow(self, title, img):
        """捕捉转场帧光流"""
        if not self.view:
            return
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, img)
        cv2.moveWindow(title, 0, 0)
        cv2.resizeWindow(title, 960, 270)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_scene(self, _img1, _img2, add_diff=False, use_diff=-1.0) -> bool:
        """
                检查当前img1是否是转场
                :param use_diff: 使用已计算出的absdiff
                :param _img2:
                :param _img1:
                :param add_diff: 仅添加absdiff到计算队列中
                :return: 是转场则返回真
                """
        img1 = _img1.copy()
        img2 = _img2.copy()

        if self.no_scdet:
            return False

        if use_diff != -1:
            diff = use_diff
        else:
            diff = self.utils.get_norm_img_diff(img1, img2)

        if self.use_fixed_scdet:
            if diff < self.dead_thres:
                return False
            else:
                self.scdet_cnt += 1
                return True

        self.img1 = img1
        self.img2 = img2

        """检测开头转场"""
        if diff < 0.001:
            """000000"""
            if self.utils.check_pure_img(img1):
                self.black_scene_queue.append(0)
            return False
        elif np.mean(self.black_scene_queue) == 0:
            """检测到00000001"""
            self.black_scene_queue.clear()
            self.scdet_cnt += 1
            self.see_result(f"absdiff: {diff:.3f}, Pure Scene Alarm, cnt: {self.scdet_cnt}")
            self.flow_img = img1
            self.save_flow()
            return True

        flow_cnt, flow = self.utils.get_norm_img_flow(img1, img2, flow_thres=self.scdet_flow)

        self.absdiff_queue.append(diff)
        self.flow_img = flow

        if len(self.flow_queue) < self.scene_queue_len or add_diff or self.utils.check_pure_img(img1):
            """检测到纯色图片，那么下一帧大概率可以被识别为转场"""
            if flow_cnt > 0:
                self.flow_queue.append(flow_cnt)
            return False

        if flow_cnt == 0:
            return False

        """Judge"""
        return self.__judge_mean(flow_cnt, diff, flow)

    def update_scene_status(self, recent_scene, scene_type: str):
        """更新转场检测状态"""
        self.scedet_info[scene_type] += 1
        if scene_type == "scene":
            self.scedet_info["recent_scene"] = recent_scene

    def get_scene_status(self):
        return self.scedet_info


class SteamValidation:
    def CheckSteamAuth(self):
        if self.is_steam:
            return 0
        steam_64id = self.steamworks.Users.GetSteamID()
        valid_response = self.steamworks.Users.GetAuthSessionTicket()
        self.logger.info(f'Steam User Logged on as {steam_64id}, auth: {valid_response}')
        return valid_response

    def CheckProDLC(self) -> bool:
        if not self.is_steam:
            return False
        purchase_pro = self.steamworks.Apps.IsDLCInstalled(ArgumentManager.pro_dlc_id)
        self.logger.info(f'Steam User Purchase Pro DLC Status: {purchase_pro}')
        return purchase_pro

    def __init__(self, is_steam, logger=None):
        """
        Whether use steam for validation
        :param is_steam:
        """
        original_cwd = os.getcwd()
        self.is_steam = is_steam
        if logger is None:
            self.logger = Tools().get_logger(__file__, "")
        else:
            self.logger = logger
        self.steamworks = None
        self.steam_valid = True
        self.steam_error = ""
        if self.is_steam:
            try:
                self.steamworks = STEAMWORKS(ArgumentManager.app_id)
                self.steamworks.initialize()  # This method has to be called in order for the wrapper to become functional!
            except Exception:
                self.steam_valid = False
                self.steam_error = traceback.format_exc(limit=ArgumentManager.traceback_limit)
        os.chdir(original_cwd)


if __name__ == "__main__":
    u = Tools()
    cp = DefaultConfigParser(allow_no_value=True)
    cp.read(r"D:\60-fps-Project\arXiv2020-RIFE-main\release\SVFI.Ft.RIFE_GUI.release.v6.2.2.A\RIFE_GUI.ini",
            encoding='utf-8')
    print(cp.get("General", "UseCUDAButton=true", 6))
    print(u.clean_parsed_config(dict(cp.items("General"))))
