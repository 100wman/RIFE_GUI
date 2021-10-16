# coding: utf-8
import argparse
import datetime
import math
import os
import re
import shlex
import sys
import threading
import time
import traceback
from queue import Queue

import cv2
import numpy as np
import psutil
import tqdm

from Utils.utils import STEAMWORKS, SteamUtils, ArgumentManager, appDir, DefaultConfigParser, Tools, VideoInfo, \
    SupportFormat, OverTimeReminderBearer, ImageRead, ImageWrite, TransitionDetection_ST, \
    VideoFrameInterpolation, Hdr10PlusProcesser, DoviProcesser, EULAWriter, utils_overtime_reminder_bearer, \
    SuperResolution, overtime_reminder_deco
from skvideo.io import FFmpegWriter, FFmpegReader, EnccWriter, SVTWriter
from steamworks.exceptions import *

if ArgumentManager.is_steam:
    try:
        _steamworks = STEAMWORKS(ArgumentManager.app_id)
    except:
        pass

print(f"INFO - ONE LINE SHOT ARGS {ArgumentManager.ols_version} {datetime.date.today()}")
# TODO Fix up SVT-HEVC

"""设置环境路径"""
os.chdir(appDir)
sys.path.append(appDir)

"""Parse Args"""
parser = argparse.ArgumentParser(prog="#### SVFI CLI tool by Jeanna ####",
                                 description='Interpolation for long video/imgs footage')
basic_parser = parser.add_argument_group(title="Basic Settings, Necessary")
basic_parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                          help="原视频/图片序列文件夹路径")
basic_parser.add_argument("-c", '--config', dest='config', type=str, required=True, help="配置文件路径")
basic_parser.add_argument("-t", '--task-id', dest='task_id', type=str, required=True, help="任务id")
basic_parser.add_argument('--concat-only', dest='concat_only', action='store_true', help='只执行合并已有区块操作')
basic_parser.add_argument('--extract-only', dest='extract_only', action='store_true', help='只执行拆帧操作')
basic_parser.add_argument('--render-only', dest='render_only', action='store_true', help='只执行渲染操作')

"""Clean Args"""
args_read = parser.parse_args()
global_config_parser = DefaultConfigParser(allow_no_value=True)  # 把SVFI GUI传来的参数格式化
global_config_parser.read(args_read.config, encoding='utf-8')
global_config_parser_items = dict(global_config_parser.items("General"))
global_args = Tools.clean_parsed_config(global_config_parser_items)
global_args.update(vars(args_read))  # update -i -o -c，将命令行参数更新到config生成的字典

"""Set Global Logger"""
logger = Tools.get_logger('TMP', '')


class TaskArgumentManager(ArgumentManager):
    """
        For OLS's current input's arguments validation
    """

    def __init__(self, _args: dict):
        super().__init__(_args)
        global logger
        self.interp_times = 0
        self.max_frame_cnt = 10 ** 10
        self.all_frames_cnt = 0
        self.frames_queue_len = 0
        self.dup_skip_limit = 0
        self.input_ext = ".mp4"
        self.output_ext = ".mp4"
        self.task_info = {"chunk_cnt": 0, "render": 0,
                          "rife_now_frame": 0,
                          "recent_scene": 0, "scene_cnt": 0,
                          "decode_process_time": 0,
                          "rife_task_acquire_time": 0,
                          "rife_process_time": 0,
                          "rife_queue_len": 0,
                          "sr_now_frame": 0,
                          "sr_task_acquire_time": 0,
                          "sr_process_time": 0,
                          "sr_queue_len": 0, }  # 有关任务的实时信息

        self.__validate_io_path()
        self.__initiate_logger()
        self.__set_ffmpeg_path()
        self.video_info_instance = VideoInfo(input_file=self.input, logger=logger, project_dir=self.project_dir,
                                             interp_exp=self.rife_exp)
        self.__update_hdr_mode()
        self.__update_io_fps()
        self.__update_frames_cnt()
        self.__update_frame_size()
        self.__update_task_queue_size_by_memory()
        self.__update_io_ext()

        """Check Initiation Info"""
        logger.info(
            f"Check Interpolation Source: "
            f"FPS: {self.input_fps} -> {self.target_fps}, FRAMES_CNT: {self.all_frames_cnt}, "
            f"INTERP_TIMES: {self.interp_times}, "
            f"HDR: {self.hdr_mode}, FRAME_SIZE: {self.frame_size}, QUEUE_LEN: {self.frames_queue_len}, "
            f"INPUT_EXT: {self.input_ext}, OUTPUT_EXT: {self.output_ext}")

        self.main_error = list()

    def __update_io_ext(self):
        """update extension"""
        self.input_ext = os.path.splitext(self.input)[1] if os.path.isfile(self.input) else ""
        self.input_ext = self.input_ext.lower()
        if not self.output_ext.startswith('.'):
            self.output_ext = "." + self.output_ext
        if "ProRes" in self.render_encoder and not self.is_img_output:
            self.output_ext = ".mov"
        if self.is_img_output and self.output_ext not in SupportFormat.img_outputs:
            self.output_ext = ".png"

    def __update_io_fps(self):
        """
        Update io fps and interp times
        :return:
        """
        """Set input and target(output) fps"""
        if not self.is_img_input:  # 输入不是文件夹，使用检测到的帧率
            self.input_fps = self.video_info_instance.fps
        elif not self.input_fps:  # 输入是文件夹，使用用户的输入帧率; 用户有毒，未发现有效的输入帧率
            raise OSError("Not Find FPS, Input File is not valid")
        if not self.target_fps:  # 未找到用户的输出帧率
            self.target_fps = (2 ** self.rife_exp) * self.input_fps  # default
        if self.is_img_input:  # 图片序列输入，不保留音频（也无音频可保留
            self.is_save_audio = False

        """Set interpolation exp related to hdr mode"""
        self.interp_times = round(self.target_fps / self.input_fps)
        if self.hdr_mode == 3 or (
                self.hdr_mode == 2 and self.video_info_instance.getInputHdr10PlusMetadata() is not None):
            """DoVi or Valid HDR10 Metadata Detected, change target fps"""
            self.target_fps = self.interp_times * self.input_fps

    def __update_task_queue_size_by_memory(self):
        """Guess Memory and Fix Resolution"""
        if self.use_manual_buffer:
            # 手动指定内存占用量
            free_mem = self.manual_buffer_size * 1024
        else:
            mem = psutil.virtual_memory()
            free_mem = round(mem.free / 1024 / 1024)
        self.frames_queue_len = round(free_mem / (sys.getsizeof(
            np.random.rand(3, self.frame_size[0], self.frame_size[1])) / 1024 / 1024))
        if not self.use_manual_buffer:
            self.frames_queue_len = int(max(10.0, self.frames_queue_len))
        self.dup_skip_limit = int(0.5 * self.input_fps) + 1  # 当前跳过的帧计数超过这个值，将结束当前判断循环
        logger.info(f"Update QLen to {self.frames_queue_len}, dup_skip_limit to {self.dup_skip_limit}")

    def __update_frame_size(self):
        """规整化输出输入分辨率"""
        self.frame_size = (round(self.video_info_instance.frames_size[0]),
                           round(self.video_info_instance.frames_size[1]))

    def __update_frames_cnt(self):
        """Update All Frames Count"""
        self.all_frames_cnt = abs(int(self.video_info_instance.duration * self.target_fps))
        if self.all_frames_cnt > self.max_frame_cnt:
            raise OSError(f"SVFI can't afford input exceeding {self.max_frame_cnt} frames")

    def __update_hdr_mode(self):
        if self.hdr_mode == 0:  # Auto
            logger.info(f"Auto HDR Mode, Set HDR mode to {self.video_info_instance.hdr_mode}")
            self.hdr_mode = self.video_info_instance.hdr_mode
            # no hdr at -1, 0 checked and None, 1 hdr, 2 hdr10, 3 DV, 4 HLG
            # hdr_check_status indicates the final process mode for (hdr) input

    def __set_ffmpeg_path(self):
        """Set FFmpeg"""
        self.ffmpeg = "ffmpeg"

    def __validate_io_path(self):
        if not len(self.input):
            raise OSError("Input Path is empty")
        if not len(self.output_dir):
            """未填写输出文件夹"""
            self.output_dir = os.path.dirname(self.input)
        if os.path.isfile(self.output_dir):
            self.output_dir = os.path.dirname(self.output_dir)

        self.project_name = f"{Tools.get_filename(self.input)}_{self.task_id}"
        self.project_dir = os.path.join(self.output_dir, self.project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        sys.path.append(self.project_dir)

        """Extract Only Mode"""
        if self.extract_only and self.output_ext not in SupportFormat.img_outputs:
            # TODO Extract only mode Be removed soon
            self.is_img_output = True
            self.output_ext = ".png"
            logger.warning("Auto change output extension to png")

        """Check Img IO status"""
        if self.is_img_output:
            self.output_dir = os.path.join(self.output_dir, self.project_name)
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isfile(self.input):
            self.is_img_input = True

    def __initiate_logger(self):
        """Set Global Logger"""
        global logger
        logger = Tools.get_logger("CLI", self.project_dir, debug=self.debug)
        logger.info(f"Initial New Interpolation Project: project_dir: %s, INPUT_FILEPATH: %s", self.project_dir,
                    self.input)

    def update_task_info(self, update_dict: dict):
        self.task_info.update(update_dict)

    def get_main_error(self):
        if not len(self.main_error):
            return None
        else:
            return self.main_error[-1]

    def save_main_error(self, e: Exception):
        self.main_error.append(e)


class SteamFlow(SteamUtils):
    def __init__(self, _args: TaskArgumentManager):
        self.ARGS = _args
        super().__init__(self.ARGS.is_steam, logger)
        self.kill = False
        pass

    def steam_update_achv(self, output_path):
        """
        Update Steam Achievement
        :return:
        """
        if not self.ARGS.is_steam or self.kill:
            """If encountered serious error in the process, end steam update"""
            return
        if self.ARGS.concat_only or self.ARGS.render_only or self.ARGS.extract_only:
            return
        """Get Stat"""
        STAT_INT_FINISHED_CNT = self.GetStat("STAT_INT_FINISHED_CNT", int)
        STAT_FLOAT_FINISHED_MINUTE = self.GetStat("STAT_FLOAT_FINISHED_MIN", float)

        """Update Stat"""
        STAT_INT_FINISHED_CNT += 1
        reply = self.SetStat("STAT_INT_FINISHED_CNT", STAT_INT_FINISHED_CNT)
        if self.ARGS.all_frames_cnt >= 0:
            """Update Mission Process Time only in interpolation"""
            STAT_FLOAT_FINISHED_MINUTE += self.ARGS.all_frames_cnt / self.ARGS.target_fps / 60
            reply = self.SetStat("STAT_FLOAT_FINISHED_MIN", round(STAT_FLOAT_FINISHED_MINUTE, 2))

        """Get ACHV"""
        ACHV_Task_Frozen = self.GetAchv("ACHV_Task_Frozen")
        ACHV_Task_Cruella = self.GetAchv("ACHV_Task_Cruella")
        ACHV_Task_Suzumiya = self.GetAchv("ACHV_Task_Suzumiya")
        ACHV_Task_1000M = self.GetAchv("ACHV_Task_1000M")
        ACHV_Task_10 = self.GetAchv("ACHV_Task_10")
        ACHV_Task_50 = self.GetAchv("ACHV_Task_50")

        """Update ACHV"""
        if 'Frozen' in output_path and not ACHV_Task_Frozen:
            reply = self.SetAchv("ACHV_Task_Frozen")
        if 'Cruella' in output_path and not ACHV_Task_Cruella:
            reply = self.SetAchv("ACHV_Task_Cruella")
        if any([i in output_path for i in ['Suzumiya', 'Haruhi', '涼宮', '涼宮ハルヒの憂鬱', '涼宮ハルヒの消失', '凉宫春日']]) \
                and not ACHV_Task_Suzumiya:
            reply = self.SetAchv("ACHV_Task_Suzumiya")
        if STAT_INT_FINISHED_CNT > 10 and not ACHV_Task_10:
            reply = self.SetAchv("ACHV_Task_10")
        if STAT_INT_FINISHED_CNT > 50 and not ACHV_Task_50:
            reply = self.SetAchv("ACHV_Task_50")
        if STAT_FLOAT_FINISHED_MINUTE > 1000 and not ACHV_Task_1000M:
            reply = self.SetAchv("ACHV_Task_1000M")
        self.Store()


class IOFlow(threading.Thread):
    def __init__(self, __args: TaskArgumentManager):
        super().__init__()
        self.ARGS = __args
        self.initiation_event = threading.Event()
        self.initiation_event.clear()
        self.reminder_bearer = OverTimeReminderBearer()
        self._kill = False
        self.task_done = False

    def _release_initiation(self):
        self.initiation_event.set()

    def acquire_initiation_clock(self):
        if not self.is_alive():
            self.initiation_event.set()
        self.initiation_event.wait()

    def _get_color_info_dict(self):
        color_tag_map = {'-color_range': 'color_range',
                         '-color_primaries': 'color_primaries',
                         '-colorspace': 'color_space', '-color_trc': 'color_transfer'}
        output_dict = {}
        for ct in color_tag_map:
            output_dict.update({ct: self.ARGS.video_info_instance.video_info[color_tag_map[ct]]})
        return output_dict

    def check_chunk(self):
        """
        Get Chunk Start
        :param: del_chunk: delete all chunks existed
        :return: chunk, start_frame
        """
        if self.ARGS.is_img_output:
            """IMG OUTPUT"""
            img_writer = ImageWrite(logger, folder=self.ARGS.output_dir, start_frame=self.ARGS.interp_start,
                                    output_ext=self.ARGS.output_ext, is_tool=True)
            last_img = img_writer.get_write_start_frame()
            if self.ARGS.interp_start not in [-1, ]:
                return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)  # Manually Prioritized
            if last_img == 0:
                return 1, 0

        if self.ARGS.interp_start != -1 or self.ARGS.output_chunk_cnt != -1:
            return int(self.ARGS.output_chunk_cnt), int(self.ARGS.interp_start)

        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(self.ARGS.project_dir)
        if not len(chunk_paths):
            return 1, 0
        return chunk_cnt + 1, last_frame + 1

    def kill(self):
        self._kill = True
        self.reminder_bearer.terminate_all()

    def _task_done(self):
        self.task_done = True

    def is_task_done(self):
        """
        Only available for specific task
        :return:
        """
        return self.task_done


class ReadFlow(IOFlow):

    def __init__(self, _args: TaskArgumentManager, _output_queue: Queue):
        super().__init__(_args)
        self.name = 'Reader'
        self._output_queue = _output_queue
        self.scene_detection = TransitionDetection_ST(self.ARGS.project_dir, int(0.3 * self.ARGS.input_fps),
                                                      scdet_threshold=self.ARGS.scdet_threshold,
                                                      no_scdet=self.ARGS.is_no_scdet,
                                                      use_fixed_scdet=self.ARGS.use_scdet_fixed,
                                                      fixed_max_scdet=self.ARGS.scdet_fixed_max,
                                                      scdet_output=self.ARGS.is_scdet_output)
        self.vfi_core = VideoFrameInterpolation(self.ARGS)

    def __crop(self, img):
        """
        Crop using self.crop parameters
        :param img:
        :return:
        """
        if img is None or not all(self.ARGS.crop_param):
            return img

        h, w, _ = img.shape
        cw, ch = self.ARGS.crop_param
        if cw > w or ch > h:
            """奇怪的黑边参数，不予以处理"""
            return img
        return img[ch:h - ch, cw:w - cw]

    def __input_check(self, dedup=False):
        """
        perform input availability check and return generator of frames
        :return: chunk_cnt, start_frame, videogen, videogen_check
        """
        _debug = False
        chunk_cnt, start_frame = self.check_chunk()  # start_frame = 0
        logger.info("Resuming Video Frames...")

        """Get Frames to interpolate"""
        reminder_id = self.reminder_bearer.generate_reminder(300, logger, "Decode Input",
                                                             "Please consider terminate current process manually, check input arguments and restart. It's normal to wait for at least 30 minutes for 4K input when performing resume of workflow")
        videogen = self.__generate_frame_reader(start_frame).nextFrame()
        videogen_check = None
        if dedup:
            videogen_check = self.__generate_frame_reader(start_frame, frame_check=True).nextFrame()
        videogen_available_check = self.__generate_frame_reader(start_frame, frame_check=True).nextFrame()

        check_img1 = self.__crop(Tools.gen_next(videogen_available_check))
        self.reminder_bearer.terminate_reminder(reminder_id)
        videogen_available_check.close()
        if check_img1 is None:
            main_error = OSError(
                f"Input file is not available: {self.ARGS.input}, is img input: {self.ARGS.is_img_input},"
                f"Please Check Your Input Settings"
                f"(Start Chunk, Start Frame, Start Point, Start Frame)")
            self.ARGS.save_main_error(main_error)
            raise main_error
        return chunk_cnt, start_frame, videogen, videogen_check

    def __generate_frame_reader(self, start_frame=0, frame_check=False):
        """
        输入帧迭代器
        :return:
        """
        """If input is sequence of frames"""
        if self.ARGS.is_img_input:
            resize_param = self.ARGS.resize_param
            if self.ARGS.use_sr:
                resize_param = self.ARGS.transfer_param

            img_reader = ImageRead(logger, folder=self.ARGS.input, start_frame=self.ARGS.interp_start,
                                   exp=self.ARGS.rife_exp, resize=resize_param, )
            self.ARGS.all_frames_cnt = img_reader.get_frames_cnt()
            logger.info(f"Img Input, update frames count to {self.ARGS.all_frames_cnt}")
            return img_reader

        """If input is a video"""
        input_dict = {"-vsync": "0", }
        if self.ARGS.use_hwaccel_decode:
            input_dict.update({"-hwaccel": "auto"})

        if self.ARGS.input_start_point or self.ARGS.input_end_point:
            """任意时段任务"""
            time_fmt = "%H:%M:%S"
            start_point = datetime.datetime.strptime("00:00:00", time_fmt)
            end_point = datetime.datetime.strptime("00:00:00", time_fmt)
            if self.ARGS.input_start_point is not None:
                start_point = datetime.datetime.strptime(self.ARGS.input_start_point, time_fmt) - start_point
                input_dict.update({"-ss": self.ARGS.input_start_point})
            else:
                start_point = start_point - start_point
            if self.ARGS.input_end_point is not None:
                end_point = datetime.datetime.strptime(self.ARGS.input_end_point, time_fmt) - end_point
                input_dict.update({"-to": self.ARGS.input_end_point})
            elif self.ARGS.video_info_instance.duration:
                # no need to care about img input
                end_point = datetime.datetime.fromtimestamp(
                    self.ARGS.video_info_instance.duration) - datetime.datetime.fromtimestamp(0.0)
            else:
                end_point = end_point - end_point

            if end_point > start_point:
                start_frame = -1
                clip_duration = end_point - start_point
                clip_fps = self.ARGS.target_fps
                self.ARGS.all_frames_cnt = round(clip_duration.total_seconds() * clip_fps)
                logger.info(
                    f"Update Input Range: in {self.ARGS.input_start_point} -> out {self.ARGS.input_end_point}, "
                    f"all_frames_cnt -> {self.ARGS.all_frames_cnt}")
            else:
                if '-ss' in input_dict:
                    input_dict.pop('-ss')
                if '-to' in input_dict:
                    input_dict.pop('-to')
                logger.warning(
                    f"Invalid Input Section, change to original course")
        else:
            logger.info(f"Input Time Section is original course")

        output_dict = {
            "-vframes": str(10 ** 10), }  # use read frames cnt to avoid ffprobe, fuck

        output_dict.update(self._get_color_info_dict())

        vf_args = f"copy"
        if start_frame not in [-1, 0]:
            # not start from the beginning
            if self.ARGS.risk_resume_mode:
                """Quick Locate"""
                input_dict.update({"-ss": f"{start_frame / self.ARGS.target_fps:.3f}"})
            else:
                vf_args += f",trim=start={start_frame / self.ARGS.target_fps:.3f}"

        if self.ARGS.use_deinterlace:
            vf_args += f",yadif=parity=auto"

        if frame_check:
            """用以一拍二一拍N除重模式的预处理"""
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": f"300x300"})
        else:
            if not self.ARGS.use_sr:
                """直接用最终输出分辨率"""
                if self.ARGS.frame_size != self.ARGS.resize_param and all(self.ARGS.resize_param):
                    output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                        "-s": f"{self.ARGS.resize_width}x{self.ARGS.resize_height}"})
            else:
                """超分"""
                if self.ARGS.frame_size != self.ARGS.transfer_param and all(self.ARGS.transfer_param):
                    output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                        "-s": f"{self.ARGS.transfer_width}x{self.ARGS.transfer_height}"})

        """Quick Extraction"""
        if not self.ARGS.is_quick_extract:
            vf_args += f",format=yuv444p10le,zscale=matrixin=input:chromal=input:cin=input,format=rgb48be,format=rgb24"

        vf_args += f",minterpolate=fps={self.ARGS.target_fps}:mi_mode=dup"

        """Update video filters"""
        output_dict["-vf"] = vf_args
        logger.debug(f"reader: {input_dict} {output_dict}")
        return FFmpegReader(filename=self.ARGS.input, inputdict=input_dict, outputdict=output_dict)

    def __run_rest(self, run_time: float):
        rest_exp = 3600
        if self.ARGS.multi_task_rest and self.ARGS.multi_task_rest_interval and \
                time.time() - run_time > self.ARGS.multi_task_rest_interval * rest_exp:
            logger.info(
                f"\n\n INFO - Exceed Run Interval {self.ARGS.multi_task_rest_interval} hour. Time to Rest for 5 minutes!")
            time.sleep(600)
            return time.time()
        return run_time

    def remove_duplicate_frames(self, videogen_check: FFmpegReader.nextFrame, init=False) -> (list, list, dict):
        """
        获得新除重预处理帧数序列
        :param init: 第一次重复帧
        :param videogen_check:
        :return:
        """
        flow_dict = dict()
        canny_dict = dict()
        predict_dict = dict()
        resize_param = (40, 40)

        def get_img(i0):
            if i0 in check_frame_data:
                return check_frame_data[i0]
            else:
                return None

        def sobel(src):
            src = cv2.GaussianBlur(src, (3, 3), 0)
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, -1, 3, 0, ksize=5)
            grad_y = cv2.Sobel(gray, -1, 0, 3, ksize=5)
            return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        def calc_flow_distance(pos0: int, pos1: int, _use_flow=True):
            if not _use_flow:
                return diff_canny(pos0, pos1)
            if (pos0, pos1) in flow_dict:
                return flow_dict[(pos0, pos1)]
            if (pos1, pos0) in flow_dict:
                return flow_dict[(pos1, pos0)]

            prev_gray = cv2.cvtColor(cv2.resize(get_img(pos0), resize_param), cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(cv2.resize(get_img(pos1), resize_param), cv2.COLOR_BGR2GRAY)
            flow0 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow1 = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray,
                                                 flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                                 winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
            flow = (flow0 - flow1) / 2
            _x = flow[:, :, 0]
            _y = flow[:, :, 1]
            dis = np.linalg.norm(_x) + np.linalg.norm(_y)
            flow_dict[(pos0, pos1)] = dis
            return dis

        def diff_canny(pos0, pos1):
            if (pos0, pos1) in canny_dict:
                return canny_dict[(pos0, pos1)]
            if (pos1, pos0) in canny_dict:
                return canny_dict[(pos1, pos0)]
            img0, img1 = get_img(pos0), get_img(pos1)
            if self.ARGS.use_dedup_sobel:
                img0, img1 = sobel(img0), sobel(img1)
            canny_diff = cv2.Canny(cv2.absdiff(img0, img1), 100, 200).mean()
            canny_dict[(pos0, pos1)] = canny_diff
            return canny_diff

        def predict_scale(pos0, pos1):
            if (pos0, pos1) in predict_dict:
                return predict_dict[(pos0, pos1)]
            if (pos1, pos0) in predict_dict:
                return predict_dict[(pos1, pos0)]

            w, h, _ = get_img(pos0).shape
            diff = cv2.Canny(cv2.absdiff(get_img(pos0), get_img(pos0)), 100, 200)
            mask = np.where(diff != 0)
            try:
                xmin = min(list(mask)[0])
            except:
                xmin = 0
            try:
                xmax = max(list(mask)[0]) + 1
            except:
                xmax = w
            try:
                ymin = min(list(mask)[1])
            except:
                ymin = 0
            try:
                ymax = max(list(mask)[1]) + 1
            except:
                ymax = h
            W = xmax - xmin
            H = ymax - ymin
            S0 = w * h
            S1 = W * H
            prediction = -2 * (S1 / S0) + 3
            predict_dict[(pos0, pos1)] = prediction
            return prediction

        use_flow = True
        check_queue_size = max(self.ARGS.frames_queue_len, 200)  # 预处理长度，非重复帧
        check_frame_list = list()  # 采样图片帧数序列,key ~ LabData
        scene_frame_list = list()  # 转场图片帧数序列,key,和check_frame_list同步
        check_frame_data = dict()  # 用于判断的采样图片数据
        if init:
            logger.info("Initiating Duplicated Frames Removal Process...This might take some time")
            pbar = tqdm.tqdm(total=check_queue_size, unit="frames")
        else:
            pbar = None
        """
            check_frame_list contains key, check_frame_data contains (key, frame_data)
        """
        check_frame_cnt = -1
        while len(check_frame_list) < check_queue_size:
            check_frame_cnt += 1
            check_frame = Tools.gen_next(videogen_check)
            if check_frame is None:
                break
            if len(check_frame_list):  # len>1
                diff_result = Tools.get_norm_img_diff(check_frame_data[check_frame_list[-1]], check_frame)
                if diff_result < 0.001:
                    continue
            if init:
                pbar.update(1)
                pbar.set_description(
                    f"Process at Extract Frame {check_frame_cnt}")
            check_frame_data[check_frame_cnt] = check_frame
            check_frame_list.append(check_frame_cnt)  # key list
        if not len(check_frame_list):
            if init:
                pbar.close()
            return [], [], {}

        if init:
            pbar.close()
            pbar = tqdm.tqdm(total=len(check_frame_list), unit="frames")

        """Scene Batch Detection"""
        for i in range(len(check_frame_list) - 1):
            if init:
                pbar.update(1)
                pbar.set_description(f"Process at Scene Detect Frame {i}")
            i1 = check_frame_data[check_frame_list[i]]
            i2 = check_frame_data[check_frame_list[i + 1]]
            result = self.scene_detection.check_scene(i1, i2)
            if result:
                scene_frame_list.append(check_frame_list[i + 1])  # at i find scene

        if init:
            pbar.close()
            logger.info("Start Remove First Batch of Duplicated Frames")

        max_epoch = self.ARGS.remove_dup_mode  # 一直去除到一拍N，N为max_epoch，默认去除一拍二
        opt = []  # 已经被标记，识别的帧
        for queue_size, _ in enumerate(range(1, max_epoch), start=4):
            Icount = queue_size - 1  # 输入帧数
            Current = []  # 该轮被标记的帧
            i = 1
            try:
                while i < len(check_frame_list) - Icount:
                    c = [check_frame_list[p + i] for p in range(queue_size)]  # 读取queue_size帧图像 ~ 对应check_frame_list中的帧号
                    first_frame = c[0]
                    last_frame = c[-1]
                    count = 0
                    for step in range(1, queue_size - 2):
                        pos = 1
                        while pos + step <= queue_size - 2:
                            m0 = c[pos]
                            m1 = c[pos + step]
                            d0 = calc_flow_distance(first_frame, m0, use_flow)
                            d1 = calc_flow_distance(m0, m1, use_flow)
                            d2 = calc_flow_distance(m1, last_frame, use_flow)
                            value_scale = predict_scale(m0, m1)
                            if value_scale * d1 < d0 and value_scale * d1 < d2:
                                count += 1
                            pos += 1
                    if count == (queue_size * (queue_size - 5) + 6) / 2:
                        Current.append(i)  # 加入标记序号
                        i += queue_size - 3
                    i += 1
            except:
                logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            for x in Current:
                if x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
                    for t in range(queue_size - 3):
                        opt.append(t + x + 1)
        delgen = sorted(set(opt))  # 需要删除的帧
        for d in delgen:
            if check_frame_list[d] not in scene_frame_list:
                check_frame_list[d] = -1

        max_key = np.max(list(check_frame_data.keys()))
        if max_key not in check_frame_list:
            check_frame_list.append(max_key)
        if 0 not in check_frame_list:
            check_frame_list.insert(0, 0)
        check_frame_list = [i for i in check_frame_list if i > -1]
        return check_frame_list, scene_frame_list, check_frame_data

    # @profile
    def rife_run(self):
        """
        Go through all procedures to produce interpolation result in dedup mode
        :return:
        """

        logger.info("Activate Remove Duplicate Frames Mode")
        chunk_cnt, now_frame_key, videogen, videogen_check = self.__input_check(dedup=True)
        logger.info("Loaded Input Frames")
        is_end = False

        """Start Process"""
        run_time = time.time()
        first_run = True
        while True:
            if is_end:
                break

            if self._kill:
                logger.critical("Reader Thread Killed")
                break

            run_time = self.__run_rest(run_time)
            decode_time = time.time()
            check_frame_list, scene_frame_list, input_frame_data = self.remove_duplicate_frames(videogen_check,
                                                                                                init=first_run)
            input_frame_data = dict(input_frame_data)
            first_run = False
            self._release_initiation()
            if not len(check_frame_list):
                while True:
                    img1 = self.__crop(Tools.gen_next(videogen))
                    if img1 is None:
                        is_end = True
                        self.__feed_to_rife(now_frame_key, img1, img1, n=0,
                                            is_end=is_end)
                        break
                    self.__feed_to_rife(now_frame_key, img1, img1, n=0)
                break

            else:
                img0 = self.__crop(Tools.gen_next(videogen))
                img1 = img0.copy()
                last_frame_key = check_frame_list[0]
                now_a_key = last_frame_key
                for frame_cnt in range(1, len(check_frame_list)):
                    now_b_key = check_frame_list[frame_cnt]
                    img1 = img0.copy()
                    """A - Interpolate -> B"""
                    while True:
                        last_possible_scene = img1
                        if now_a_key != now_b_key:
                            img1 = self.__crop(Tools.gen_next(videogen))
                            now_a_key += 1
                        else:
                            break
                    now_frame_key = now_b_key
                    self.ARGS.update_task_info({"read_now_frame": now_frame_key})
                    if now_frame_key in scene_frame_list:
                        self.scene_detection.update_scene_status(now_frame_key, "scene")
                        potential_key = now_frame_key - 1
                        if potential_key > 0 and potential_key in input_frame_data:
                            before_img = last_possible_scene
                        else:
                            before_img = img0

                        # Scene Review, should be annoted
                        # title = f"try:"
                        # comp_stack = np.hstack((img0, before_img, img1))
                        # comp_stack = cv2.resize(comp_stack, (1440, 270))
                        # cv2.imshow(title, cv2.cvtColor(comp_stack, cv2.COLOR_BGR2RGB))
                        # cv2.moveWindow(title, 0, 0)
                        # cv2.resizeWindow(title, 1440, 270)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        if frame_cnt < 1:
                            self.__feed_to_rife(now_frame_key, img0, img0, n=0,
                                                is_end=is_end)
                        elif self.ARGS.is_scdet_mix:
                            self.__feed_to_rife(now_frame_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                                add_scene=True,
                                                is_end=is_end)
                        else:
                            self.__feed_to_rife(now_frame_key, img0, before_img, n=now_frame_key - last_frame_key - 2,
                                                add_scene=True,
                                                is_end=is_end)
                    else:
                        self.scene_detection.update_scene_status(now_frame_key, "normal")
                        self.__feed_to_rife(now_b_key, img0, img1, n=now_frame_key - last_frame_key - 1,
                                            is_end=is_end)
                    self.update_decode_process_time(time.time() - decode_time)
                    last_frame_key = now_frame_key
                    img0 = img1
                    self.update_scene_status()
                self.__feed_to_rife(now_frame_key, img1, img1, n=0, is_end=is_end)
                self.ARGS.update_task_info({"read_now_frame": check_frame_list[-1]})

        pass
        self._output_queue.put(None)
        videogen.close()
        videogen_check.close()

    def update_decode_process_time(self, decode_time):
        self.ARGS.update_task_info({'decode_process_time': decode_time})

    # @profile
    def rife_run_any_fps(self):
        """
        Go through all procedures to produce interpolation result in any fps mode(from a fps to b fps)
        :return:
        """

        logger.info("Activate Any FPS Mode")
        chunk_cnt, now_frame, videogen, videogen_check = self.__input_check(dedup=True)
        img1 = self.__crop(Tools.gen_next(videogen))
        logger.info("Loaded Input Frames")
        is_end = False

        """Update Interp Mode Info"""
        if self.ARGS.remove_dup_mode == 1:  # 单一模式
            self.ARGS.remove_dup_threshold = self.ARGS.remove_dup_threshold if self.ARGS.remove_dup_threshold > 0.01 else 0.01
        else:  # 0， 不去除重复帧
            self.ARGS.remove_dup_threshold = 0.001

        """Start Process"""
        run_time = time.time()
        self._release_initiation()

        while True:
            if is_end:
                break

            if self._kill:
                logger.critical("Reader Thread Killed")
                break

            run_time = self.__run_rest(run_time)
            decode_time = time.time()
            img0 = img1
            img1 = self.__crop(Tools.gen_next(videogen))

            now_frame += 1

            if img1 is None:
                is_end = True
                self.__feed_to_rife(now_frame, img0, img0, is_end=is_end)
                self.update_decode_process_time(time.time() - decode_time)
                break

            diff = Tools.get_norm_img_diff(img0, img1)
            skip = 0  # 用于记录跳过的帧数

            """Find Scene"""
            if self.scene_detection.check_scene(img0, img1, use_diff=diff):
                self.__feed_to_rife(now_frame, img0, img1, n=0,
                                    is_end=is_end)  # add img0 only, for there's no gap between img0 and img1
                self.update_decode_process_time(time.time() - decode_time)
                self.scene_detection.update_scene_status(now_frame, "scene")
                continue
            else:
                if diff < self.ARGS.remove_dup_threshold:
                    before_img = img1.copy()
                    is_scene = False
                    while diff < self.ARGS.remove_dup_threshold:
                        skip += 1
                        self.scene_detection.update_scene_status(now_frame, "dup")
                        last_frame = img1.copy()
                        img1 = self.__crop(Tools.gen_next(videogen))

                        if img1 is None:
                            img1 = last_frame
                            is_end = True
                            break

                        diff = Tools.get_norm_img_diff(img0, img1)

                        is_scene = self.scene_detection.check_scene(img0, img1, use_diff=diff)  # update scene stack
                        if is_scene:
                            break
                        if skip == self.ARGS.dup_skip_limit * self.ARGS.target_fps // self.ARGS.input_fps:
                            """超过重复帧计数限额，直接跳出"""
                            break

                    # 除去重复帧后可能im0，im1依然为转场，因为转场或大幅度运动的前一帧可以为重复帧
                    if is_scene:
                        if self.ARGS.is_scdet_mix:
                            self.__feed_to_rife(now_frame, img0, img1, n=skip, add_scene=True,
                                                is_end=is_end)
                        else:
                            self.__feed_to_rife(now_frame, img0, before_img, n=skip - 1, add_scene=True,
                                                is_end=is_end)
                            """
                            0 (1 2 3) 4[scene] => 0 (1 2) 3 4[scene] 括号内为RIFE应该生成的帧
                            """
                        self.scene_detection.update_scene_status(now_frame, "scene")
                        self.update_decode_process_time(time.time() - decode_time)

                    elif skip != 0:  # skip >= 1
                        assert skip >= 1
                        """Not Scene"""
                        self.__feed_to_rife(now_frame, img0, img1, n=skip, is_end=is_end)
                        self.scene_detection.update_scene_status(now_frame, "normal")
                        self.update_decode_process_time(time.time() - decode_time)
                    now_frame += skip
                else:
                    """normal frames"""
                    self.__feed_to_rife(now_frame, img0, img1, n=0, is_end=is_end)  # 当前模式下非重复帧间没有空隙，仅输入img0
                    self.scene_detection.update_scene_status(now_frame, "normal")
                    self.update_decode_process_time(time.time() - decode_time)

                self.ARGS.update_task_info({"read_now_frame": now_frame})
                self.update_scene_status()
            pass

        self._output_queue.put(None)  # bad way to end
        videogen.close()
        videogen_check.close()

    def __feed_to_rife(self, now_frame: int, img0, img1, n=0, exp=0, is_end=False, add_scene=False, ):
        """
        创建任务，输出到补帧任务队列消费者
        :param now_frame:当前帧数
        :param add_scene:加入转场的前一帧（避免音画不同步和转场鬼畜）
        :param img0:
        :param img1:
        :param n:要补的帧数
        :param exp:使用指定的补帧倍率（2**exp）
        :param is_end:是否是任务结束
        :return:
        """
        scale = self.ARGS.rife_scale
        if self.ARGS.use_rife_auto_scale:
            """使用动态光流"""
            if img0 is None or img1 is None:
                scale = 1.0
            else:
                scale = self.vfi_core.get_auto_scale(img0, img1)

        self._output_queue.put(
            {"now_frame": now_frame, "img0": img0, "img1": img1, "n": n, "scale": scale,
             "is_end": is_end, "add_scene": add_scene})

    def update_vfi_core(self, vfi_core: VideoFrameInterpolation):
        self.vfi_core = vfi_core

    def update_scene_status(self):
        scene_status = self.scene_detection.get_scene_status()
        update_dict = {'recent_scene': scene_status['recent_scene'], 'scene_cnt': scene_status['scene']}
        self.ARGS.update_task_info(update_dict)

    def run(self):
        try:
            if self.ARGS.remove_dup_mode in [0, 1]:
                self.rife_run_any_fps()
            else:  # 1, 2 => 去重一拍二或一拍三
                self.rife_run()
            self._task_done()
        except Exception as e:
            logger.critical("Read Thread Panicked")
            logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
            return


class RenderFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, _reader_queue: Queue):
        super().__init__(_args)
        self.name = 'Render'
        self.__ffmpeg = "ffmpeg"
        self.__hdr10_metadata_processer = Hdr10PlusProcesser(logger, self.ARGS.project_dir, self.ARGS.render_gap,
                                                             self.ARGS.interp_times,
                                                             self.ARGS.video_info_instance.getInputHdr10PlusMetadata())
        self._input_queue = _reader_queue
        self.is_audio_failed_concat = False
        self.steam_flow = None

    def __modify_hdr_params(self):
        if self.ARGS.is_img_input or self.ARGS.hdr_mode == 1:  # img input or ordinary hdr
            return

        if self.ARGS.hdr_mode == 2:
            """HDR10"""
            self.ARGS.render_hwaccel_mode = "CPU"
            if "H265" in self.ARGS.render_encoder:
                self.ARGS.render_encoder = "H265, 10bit"
            elif "H264" in self.ARGS.render_encoder:
                self.ARGS.render_encoder = "H264, 10bit"
            self.ARGS.render_encoder_preset = "medium"
        elif self.ARGS.hdr_mode == 4:
            """HLG"""
            self.ARGS.render_encoder = "H265, 10bit"
            self.ARGS.render_hwaccel_mode = "CPU"
            self.ARGS.render_encoder_preset = "medium"

    def __generate_frame_writer(self, start_frame: int, output_path: str):
        """
        渲染帧
        :param start_frame: for IMG IO, select start_frame to generate IO instance
        :param output_path:
        :return:
        """
        hdr10plus_metadata_path = self.__hdr10_metadata_processer.get_hdr10plus_metadata_path_at_point(start_frame)
        params_libx265s = {
            "fast": "high-tier=0:ref=2:rd=1:ctu=32:rect=0:amp=0:early-skip=1:fast-intra=1:b-intra=1:"
                    "rdoq-level=0:me=2:subme=3:merange=25:weightb=1:strong-intra-smoothing=0:open-gop=0:keyint=250:"
                    "min-keyint=1:rc-lookahead=25:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:"
                    "crqpoffs=-2:qcomp=0.65:deblock=-1:sao=0:repeat-headers=1",
            "8bit": "high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:"
                    "strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:"
                    "rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:"
                    "qcomp=0.65:deblock=-1:sao=0",
            "10bit": "high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:"
                     "strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:"
                     "rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:"
                     "deblock=-1:sao=0",
            "hdr10": 'high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:'
                     'strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:'
                     'rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:'
                     'deblock=-1:sao=0:'
                     'range=limited:colorprim=9:transfer=16:colormatrix=9:'
                     'master-display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)":'
                     'max-cll="1000,100":hdr10-opt=1:repeat-headers=1',
            "hdr10+": 'high-tier=0:ref=3:rd=3:rect=0:amp=0:b-intra=1:rdoq-level=2:limit-tu=4:me=3:subme=5:weightb=1:'
                      'strong-intra-smoothing=0:psy-rd=2.0:psy-rdoq=1.0:open-gop=0:keyint=250:min-keyint=1:'
                      'rc-lookahead=50:bframes=6:aq-mode=1:aq-strength=0.8:qg-size=8:cbqpoffs=-2:crqpoffs=-2:qcomp=0.65:'
                      'deblock=-1:sao=0:'
                      'range=limited:colorprim=9:transfer=16:colormatrix=9:'
                      'master-display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)":'
                      f'max-cll="1000,100":dhdr10-info="{hdr10plus_metadata_path}"'
        }

        params_libx264s = {
            "fast": "keyint=250:min-keyint=1:bframes=6:b-adapt=2:open-gop=0:ref=4:deblock='-1:-1':"
                    "rc-lookahead=30:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:me=hex:merange=16:"
                    "subme=7:psy-rd='1:0.1':mixed-refs=1:trellis=1",
            "8bit": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:deblock='-1:-1':"
                    "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                    "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0",
            "10bit": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:deblock='-1:-1':"
                     "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                     "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0",
            "hdr10": "keyint=250:min-keyint=1:bframes=8:b-adapt=2:open-gop=0:ref=12:deblock='-1:-1':"
                     "rc-lookahead=60:chroma-qp-offset=-2:aq-mode=1:aq-strength=0.8:qcomp=0.75:partitions=all:"
                     "direct=auto:me=umh:merange=24:subme=10:psy-rd='1:0.1':mixed-refs=1:trellis=2:fast-pskip=0:"
                     "range=tv:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                     "mastering-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)':"
                     "cll='1000,100'"
        }

        """If output is sequence of frames"""
        if self.ARGS.is_img_output:
            img_io = ImageWrite(logger, folder=self.ARGS.output_dir, start_frame=start_frame, exp=self.ARGS.rife_exp,
                                resize=self.ARGS.resize_param, output_ext=self.ARGS.output_ext, )
            return img_io

        """HDR Check"""
        if self.ARGS.hdr_mode != 0:
            self.__modify_hdr_params()

        """Output Video"""
        input_dict = {"-vsync": "cfr"}

        output_dict = {"-r": f"{self.ARGS.target_fps}", "-preset": self.ARGS.render_encoder_preset,
                       "-metadata": f'title="Powered By SVFI {self.ARGS.version}"'}

        output_dict.update(self._get_color_info_dict())

        if not self.ARGS.is_img_input:
            input_dict.update({"-r": f"{self.ARGS.target_fps}"})
        else:
            """Img Input"""
            input_dict.update({"-r": f"{self.ARGS.input_fps * self.ARGS.interp_times}"})

        """Slow motion design"""
        if self.ARGS.is_render_slow_motion:
            if self.ARGS.render_slow_motion_fps:
                input_dict.update({"-r": f"{self.ARGS.render_slow_motion_fps}"})
            else:
                input_dict.update({"-r": f"{self.ARGS.target_fps}"})
            output_dict.pop("-r")

        vf_args = "copy"  # debug
        output_dict.update({"-vf": vf_args})

        if self.ARGS.use_sr and all(self.ARGS.resize_param):
            output_dict.update({"-sws_flags": "lanczos+full_chroma_inp",
                                "-s": f"{self.ARGS.resize_width}x{self.ARGS.resize_height}"})

        """Assign Render Codec"""
        """CRF / Bitrate Control"""
        if self.ARGS.render_hwaccel_mode == "CPU":
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx264", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "high",
                                        "-x264-params": params_libx264s["8bit"]})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "high10",
                                        "-x264-params": params_libx264s["10bit"]})
                if 'fast' in self.ARGS.render_encoder_preset:
                    output_dict.update({"-x264-params": params_libx264s["fast"]})
                if self.ARGS.hdr_mode == 2:
                    """HDR10"""
                    output_dict.update({"-x264-params": params_libx264s["hdr10"]})
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "libx265", "-preset:v": self.ARGS.render_encoder_preset})
                if "8bit" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv420p", "-profile:v": "main",
                                        "-x265-params": params_libx265s["8bit"]})
                else:
                    """10bit"""
                    output_dict.update({"-pix_fmt": "yuv420p10", "-profile:v": "main10",
                                        "-x265-params": params_libx265s["10bit"]})
                if 'fast' in self.ARGS.render_encoder_preset:
                    output_dict.update({"-x265-params": params_libx265s["fast"]})
                if self.ARGS.hdr_mode == 2:
                    """HDR10"""
                    output_dict.update({"-x265-params": params_libx265s["hdr10"]})
                    if os.path.exists(hdr10plus_metadata_path):
                        output_dict.update({"-x265-params": params_libx265s["hdr10+"]})
            else:
                """ProRes"""
                if "-preset" in output_dict:
                    output_dict.pop("-preset")
                output_dict.update({"-c:v": "prores_ks", "-profile:v": self.ARGS.render_encoder_preset, })
                if "422" in self.ARGS.render_encoder:
                    output_dict.update({"-pix_fmt": "yuv422p10le"})
                else:
                    output_dict.update({"-pix_fmt": "yuv444p10le"})

        elif self.ARGS.render_hwaccel_mode == "NVENC":
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update(
                    {f"-g": f"{int(self.ARGS.target_fps * 3)}", "-c:v": "h264_nvenc", "-rc:v": "vbr_hq", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_nvenc", "-rc:v": "vbr_hq",
                                    f"-g": f"{int(self.ARGS.target_fps * 3)}", })

            if self.ARGS.render_encoder_preset != "loseless":
                hwacccel_preset = self.ARGS.render_hwaccel_preset
                if hwacccel_preset != "None":
                    output_dict.update({"-i_qfactor": "0.71", "-b_qfactor": "1.3", "-keyint_min": "1",
                                        f"-rc-lookahead": "120", "-forced-idr": "1", "-nonref_p": "1",
                                        "-strict_gop": "1", })
                    if hwacccel_preset == "5th":
                        output_dict.update({"-bf": "0"})
                    elif hwacccel_preset == "6th":
                        output_dict.update({"-bf": "0", "-weighted_pred": "1"})
                    elif hwacccel_preset == "7th+":
                        output_dict.update({"-bf": "4", "-temporal-aq": "1", "-b_ref_mode": "2"})
            else:
                output_dict.update({"-preset": "10", })

        elif self.ARGS.render_hwaccel_mode == "NVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "NVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                # "--chroma-qp-offset": "-2",
                "--lookahead": "16",
                "--gop-len": "250",
                "-b": "4",
                "--ref": "8",
                "--aq": "",
                "--aq-temporal": "",
                "--bref-mode": "middle"}
            if '-color_range' in output_dict:
                _output_dict.update({"--colorrange": output_dict["-color_range"]})
            if '-colorspace' in output_dict:
                _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            if '-color_trc' in output_dict:
                _output_dict.update({"--transfer": output_dict["-color_trc"]})
            if '-color_primaries' in output_dict:
                _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encoder:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high10" if "10bit" in self.ARGS.render_encoder else "high", })
            elif "H265" in self.ARGS.render_encoder:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "-b": "5"})

            if self.ARGS.hdr_mode == 2:
                """HDR10"""
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10",
                                     "--tier": "main", "-b": "5",
                                     "--max-cll": "1000,100",
                                     "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"})
                if os.path.exists(hdr10plus_metadata_path):
                    _output_dict.update({"--dhdr10-info": hdr10plus_metadata_path})
            else:
                if self.ARGS.render_encoder_preset != "loseless":
                    _output_dict.update({"--preset": self.ARGS.render_encoder_preset})
                else:
                    _output_dict.update({"--lossless": "", "--preset": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_hwaccel_mode == "QSVENCC":
            _input_dict = {  # '--avsw': '',
                'encc': "QSVENCC",
                '--fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
            }
            _output_dict = {
                "--fallback-rc": "", "--la-depth": "50", "--la-quality": "slow", "--extbrc": "", "--mbbrc": "",
                "--i-adapt": "",
                "--b-adapt": "", "--gop-len": "250", "-b": "6", "--ref": "8", "--b-pyramid": "", "--weightb": "",
                "--weightp": "", "--adapt-ltr": "",
            }
            if '-color_range' in output_dict:
                _output_dict.update({"--colorrange": output_dict["-color_range"]})
            if '-colorspace' in output_dict:
                _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            if '-color_trc' in output_dict:
                _output_dict.update({"--transfer": output_dict["-color_trc"]})
            if '-color_primaries' in output_dict:
                _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'--output-res': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"--output-depth": "10"})
            if "H264" in self.ARGS.render_encoder:
                _output_dict.update({f"-c": f"h264",
                                     "--profile": "high", "--repartition-check": "", "--trellis": "all"})
            elif "H265" in self.ARGS.render_encoder:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64", })
            if self.ARGS.hdr_mode == 2:
                _output_dict.update({"-c": "hevc",
                                     "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
                                     "--tier": "main", "--sao": "luma", "--ctu": "64",
                                     "--max-cll": "1000,100",
                                     "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"
                                     })
            _output_dict.update({"--quality": self.ARGS.render_encoder_preset})

            input_dict = _input_dict
            output_dict = _output_dict
            pass
        elif self.ARGS.render_hwaccel_mode == "SVT":
            _input_dict = {  # '--avsw': '',
                '-fps': output_dict['-r'] if '-r' in output_dict else input_dict['-r'],
                "-pix_fmt": "rgb24",
                '-n': f"{self.ARGS.render_gap}"
            }
            _output_dict = {
                "encc": "hevc", "-brr": "1", "-sharp": "1", "-b": ""
            }
            if "VP9" in self.ARGS.render_encoder:
                _output_dict = {
                    "encc": "vp9", "-tune": "0", "-b": ""
                }
            # TODO SVT Color Info
            # if '-color_range' in output_dict:
            #     _output_dict.update({"--colorrange": output_dict["-color_range"]})
            # if '-colorspace' in output_dict:
            #     _output_dict.update({"--colormatrix": output_dict["-colorspace"]})
            # if '-color_trc' in output_dict:
            #     _output_dict.update({"--transfer": output_dict["-color_trc"]})
            # if '-color_primaries' in output_dict:
            #     _output_dict.update({"--colorprim": output_dict["-color_primaries"]})

            if '-s' in output_dict:
                _output_dict.update({'-s': output_dict['-s']})
            if "10bit" in self.ARGS.render_encoder:
                _output_dict.update({"-bit-depth": "10"})
            else:
                _output_dict.update({"-bit-depth": "8"})

            preset_mapper = {"slowest": "4", "slow": "5", "fast": "7", "faster": "9"}

            if "H265" in self.ARGS.render_encoder_preset:
                _output_dict.update({"-encMode": preset_mapper[self.ARGS.render_encoder_preset]})
            elif "VP9" in self.ARGS.render_encoder_preset:
                _output_dict.update({"-enc-mode": preset_mapper[self.ARGS.render_encoder_preset]})

            # TODO SVT max cll, master display
            # if self.ARGS.hdr_mode == 2:
            #     _output_dict.update({"-c": "hevc",
            #                          "--profile": "main10" if "10bit" in self.ARGS.render_encoder else "main",
            #                          "--tier": "main", "--sao": "luma", "--ctu": "64",
            #                          "--max-cll": "1000,100",
            #                          "--master-display": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"
            #                          })

            input_dict = _input_dict
            output_dict = _output_dict
            pass

        else:
            """QSV"""
            output_dict.update({"-pix_fmt": "yuv420p"})
            if "10bit" in self.ARGS.render_encoder:
                output_dict.update({"-pix_fmt": "yuv420p10le"})
                pass
            if "H264" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "h264_qsv",
                                    "-i_qfactor": "0.75", "-b_qfactor": "1.1",
                                    f"-rc-lookahead": "120", })
            elif "H265" in self.ARGS.render_encoder:
                output_dict.update({"-c:v": "hevc_qsv",
                                    f"-g": f"{int(self.ARGS.target_fps * 3)}", "-i_qfactor": "0.75",
                                    "-b_qfactor": "1.1",
                                    f"-look_ahead": "120", })

        if "ProRes" not in self.ARGS.render_encoder and self.ARGS.render_encoder_preset != "loseless":

            if self.ARGS.render_crf and self.ARGS.use_crf:
                if self.ARGS.render_hwaccel_mode != "CPU":
                    hwaccel_mode = self.ARGS.render_hwaccel_mode
                    if hwaccel_mode == "NVENC":
                        output_dict.update({"-cq:v": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "QSV":
                        output_dict.update({"-q": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "NVENCC":
                        output_dict.update({"--vbr": "0", "--vbr-quality": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "QSVENCC":
                        output_dict.update({"--la-icq": str(self.ARGS.render_crf)})
                    elif hwaccel_mode == "SVT":
                        output_dict.update({"-q": str(self.ARGS.render_crf)})

                else:  # CPU
                    output_dict.update({"-crf": str(self.ARGS.render_crf)})

            if self.ARGS.render_bitrate and self.ARGS.use_bitrate:
                if self.ARGS.render_hwaccel_mode in ["NVENCC", "QSVENCC"]:
                    output_dict.update({"--vbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                elif self.ARGS.render_hwaccel_mode == "SVT":
                    output_dict.update({"-tbr": f'{int(self.ARGS.render_bitrate * 1024)}'})
                else:
                    output_dict.update({"-b:v": f'{self.ARGS.render_bitrate}M'})
                if self.ARGS.render_hwaccel_mode == "QSV":
                    output_dict.update({"-maxrate": "200M"})

        if self.ARGS.use_manual_encode_thread:
            if self.ARGS.render_hwaccel_mode == "NVENCC":
                output_dict.update({"--output-thread": f"{self.ARGS.use_manual_encode_thread}"})
            else:
                output_dict.update({"-threads": f"{self.ARGS.use_manual_encode_thread}"})

        logger.debug(f"writer: {output_dict}, {input_dict}")

        """Customize FFmpeg Render Command"""
        ffmpeg_customized_command = {}
        if len(self.ARGS.render_ffmpeg_customized):
            shlex_out = shlex.split(self.ARGS.render_ffmpeg_customized)
            if len(shlex_out) % 2 != 0:
                logger.warning(f"Customized FFmpeg is invalid: {self.ARGS.render_ffmpeg_customized}")
            else:
                for i in range(int(len(shlex_out) / 2)):
                    ffmpeg_customized_command.update({shlex_out[i * 2]: shlex_out[i * 2 + 1]})
        logger.debug(f"ffmpeg custom: {ffmpeg_customized_command}")
        output_dict.update(ffmpeg_customized_command)
        if self.ARGS.render_hwaccel_mode in ["NVENCC", "QSVENCC"]:
            return EnccWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)
        elif self.ARGS.render_hwaccel_mode in ["SVT"]:
            return SVTWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)
        return FFmpegWriter(filename=output_path, inputdict=input_dict, outputdict=output_dict)

    def __rename_chunk(self, chunk_from_path: str, chunk_cnt: int, start_frame: int, end_frame: int):
        """Maintain Chunk json"""
        if self.ARGS.is_img_output or self._kill:
            return
        chunk_desc_path = "chunk-{:0>3d}-{:0>8d}-{:0>8d}{}".format(chunk_cnt, start_frame, end_frame,
                                                                   self.ARGS.output_ext)
        chunk_desc_path = os.path.join(self.ARGS.project_dir, chunk_desc_path)
        if os.path.exists(chunk_desc_path):
            os.remove(chunk_desc_path)
        if os.path.exists(chunk_from_path):
            os.rename(chunk_from_path, chunk_desc_path)
        else:
            logger.warning(f"Rename Chunk Not find {chunk_from_path}")

    def __check_audio_concat(self, chunk_tmp_path: str, fail_signal=0):
        """Check Input file ext"""
        # TODO Recursive Check and check lock
        if not self.ARGS.is_save_audio or self.ARGS.get_main_error() is not None:
            return
        if self.ARGS.is_img_output:
            return
        output_ext = self.ARGS.output_ext

        concat_filepath = f"{os.path.join(self.ARGS.output_dir, 'concat_test')}" + output_ext
        map_audio = f'-i "{self.ARGS.input}" -map 0:v:0 -map 1:a:0 -map 1:s? -c:a copy -c:s copy -shortest '
        ffmpeg_command = f'{self.__ffmpeg} -hide_banner -i "{chunk_tmp_path}" {map_audio} -c:v copy ' \
                         f'{Tools.fillQuotation(concat_filepath)} -y'

        logger.info("Start Audio Concat Test")
        sp = Tools.popen(ffmpeg_command)
        sp.wait()
        if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
            if self.ARGS.input_ext in SupportFormat.vid_outputs:
                self.ARGS.output_ext = self.ARGS.input_ext
                logger.warning(f"Concat Test found unavailable output extension {self.ARGS.output_ext}, "
                               f"changed to {self.ARGS.input_ext}")
            else:
                logger.error(f"Concat Test Error, {output_ext}, empty output")
                main_error = FileExistsError(
                    "Concat Test Error, empty output detected, Please Check Your Output Extension!!!\n"
                    "e.g. mkv input should match .mkv as output extension to avoid possible concat issues")
                self.ARGS.save_main_error(main_error)
        else:
            logger.info("Audio Concat Test Success")
            os.remove(concat_filepath)

    def get_output_path(self):
        """
        Get Output Path for Process
        :return:
        """
        """Check Input file ext"""
        output_ext = self.ARGS.output_ext
        if "ProRes" in self.ARGS.render_encoder:
            output_ext = ".mov"

        output_filepath = f"{os.path.join(self.ARGS.output_dir, Tools.get_filename(self.ARGS.input))}"
        if self.ARGS.render_only:
            output_filepath += "_SVFI_Render"  # 仅渲染
        output_filepath += f"_{int(self.ARGS.target_fps)}fps"  # 补帧

        if self.ARGS.is_render_slow_motion:  # 慢动作
            output_filepath += f"_[SLM_{self.ARGS.render_slow_motion_fps}fps]"
        if self.ARGS.use_deinterlace:
            output_filepath += f"_[DI]"
        if self.ARGS.use_fast_denoise:
            output_filepath += f"_[DN]"

        if not self.ARGS.render_only:
            """RIFE"""
            if self.ARGS.use_rife_auto_scale:
                output_filepath += f"_[DS]"
            else:
                output_filepath += f"_[S-{self.ARGS.rife_scale}]"  # 全局光流尺度
            if self.ARGS.use_ncnn:
                output_filepath += "_[NCNN]"
            output_filepath += f"_[{os.path.basename(self.ARGS.rife_model_name)}]"  # 添加模型信息
            if self.ARGS.use_rife_fp16:
                output_filepath += "_[FP16]"
            if self.ARGS.is_rife_reverse:
                output_filepath += "_[RR]"
            if self.ARGS.use_rife_forward_ensemble:
                output_filepath += "_[RFE]"
            if self.ARGS.rife_tta_mode:
                output_filepath += f"_[TTA-{self.ARGS.rife_tta_mode}-{self.ARGS.rife_tta_iter}]"
            if self.ARGS.remove_dup_mode:  # 去重模式
                output_filepath += f"_[FD-{self.ARGS.remove_dup_mode}]"

        if self.ARGS.use_sr:  # 使用超分
            sr_model = os.path.splitext(self.ARGS.use_sr_model)[0]
            output_filepath += f"_[SR-{self.ARGS.use_sr_algo}-{sr_model}]"

        output_filepath += f"_{self.ARGS.task_id[-6:]}"
        output_filepath += output_ext  # 添加后缀名
        return output_filepath, output_ext

    # @profile
    @overtime_reminder_deco(300, logger, "Concat Chunks",
                            "This is normal for long footage more than 30 chunks, please wait patiently until concat is done")
    def concat_all(self):
        """
        Concat all the chunks
        :return:
        """

        os.chdir(self.ARGS.project_dir)
        concat_path = os.path.join(self.ARGS.project_dir, "concat.ini")
        logger.info("Final Round Finished, Start Concating")
        concat_list = list()

        for f in os.listdir(self.ARGS.project_dir):
            if re.match("chunk-\d+-\d+-\d+", f):
                concat_list.append(os.path.join(self.ARGS.project_dir, f))
            else:
                logger.debug(f"concat escape {f}")

        concat_list.sort(key=lambda x: int(os.path.basename(x).split('-')[2]))  # sort as start-frame

        if not len(concat_path):
            raise OSError(
                f"Could not find any chunks, the chunks could have already been concatenated or removed, please check your output folder.")

        if os.path.exists(concat_path):
            os.remove(concat_path)

        with open(concat_path, "w+", encoding="UTF-8") as w:
            for f in concat_list:
                w.write(f"file '{f}'\n")

        concat_filepath, output_ext = self.get_output_path()

        if self.ARGS.is_save_audio and not self.ARGS.is_img_input:
            audio_path = self.ARGS.input
            map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a:0 -map 1:s? -c:a copy -c:s copy '
            if self.ARGS.input_start_point or self.ARGS.input_end_point:
                map_audio = f'-i "{audio_path}" -map 0:v:0 -map 1:a:0 -c:a aac -ab 640k '
                if self.ARGS.input_end_point is not None:
                    map_audio = f'-to {self.ARGS.input_end_point} {map_audio}'
                if self.ARGS.input_start_point is not None:
                    map_audio = f'-ss {self.ARGS.input_start_point} {map_audio}'
            if self.ARGS.input_ext in ['.vob'] and self.ARGS.output_ext in ['.mkv']:
                map_audio += "-map_chapters -1 "

        else:
            map_audio = ""

        color_dict = self._get_color_info_dict()
        color_info_str = ""
        for ck, cd in color_dict.items():
            color_info_str += f" {ck} {cd}"

        ffmpeg_command = f'{self.__ffmpeg} -hide_banner -f concat -safe 0 -i "{concat_path}" {map_audio} -c:v copy ' \
                         f'{Tools.fillQuotation(concat_filepath)} -metadata title="Powered By SVFI {self.ARGS.version}" ' \
                         f'{color_info_str} ' \
                         f'-y'

        logger.debug(f"Concat command: {ffmpeg_command}")
        sp = Tools.popen(ffmpeg_command)
        sp.wait()
        logger.info(f"Concat {len(concat_list)} files to {os.path.basename(concat_filepath)}")
        if self.ARGS.hdr_mode == 3:
            self.__run_dovi(concat_filepath)
        if not os.path.exists(concat_filepath) or not os.path.getsize(concat_filepath):
            main_error = FileExistsError(
                f"Concat Error with output extension {output_ext}, empty output detected, Please Check Your Output Extension!!!\n"
                "e.g. mkv input should match .mkv as output extension to avoid possible concat issues")
            self.ARGS.save_main_error(main_error)
            raise main_error
        if self.ARGS.is_output_only:
            self.__del_existed_chunks()

        if self.steam_flow is SteamFlow:
            self.steam_flow.steam_update_achv(concat_filepath)

    def check_concat_result(self):
        concat_filepath, output_ext = self.get_output_path()
        if os.path.exists(concat_filepath):
            logger.warning("Mission Already Finished, "
                           "Jump to Dolby Vision Check")
            if self.ARGS.hdr_mode == 3:
                """Dolby Vision"""
                self.__run_dovi(concat_filepath)
            else:
                return True
        return False

    def __del_existed_chunks(self):
        chunk_paths, chunk_cnt, last_frame = Tools.get_existed_chunks(self.ARGS.project_dir)
        for f in chunk_paths:
            os.remove(os.path.join(self.ARGS.project_dir, f))

    def __run_dovi(self, concat_filepath: str):
        logger.info("Start DOVI Conversion")
        dovi_maker = DoviProcesser(concat_filepath, self.ARGS.input, self.ARGS.project_dir,
                                   self.ARGS.interp_times, logger)
        dovi_maker.run()

    def run(self):
        """
                Render thread
                :return:
        """
        concat_test_flag = True
        chunk_cnt, start_frame = self.check_chunk()
        chunk_frame_cnt = 1  # number of frames of current output chunk
        chunk_tmp_path = os.path.join(self.ARGS.project_dir, f"chunk-tmp{self.ARGS.output_ext}")
        frame_writer = self.__generate_frame_writer(start_frame, chunk_tmp_path, )  # get frame renderer

        now_frame = start_frame
        is_end = False
        frame_written = False
        self._release_initiation()
        try:
            while True:
                if self._kill:
                    if frame_written:
                        frame_writer.close()
                    logger.warning("Render thread killed, break")  # 主线程已结束，这里的锁其实没用，调试用的
                    frame_writer.close()
                    is_end = True
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, now_frame)
                    break

                frame_data = self._input_queue.get()
                if frame_data is None:
                    if frame_written:
                        frame_writer.close()
                    is_end = True
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, now_frame)
                    break

                now_frame = frame_data[0]
                frame = frame_data[1]

                if self.ARGS.use_fast_denoise:
                    frame = cv2.fastNlMeansDenoising(frame)

                reminder_id = self.reminder_bearer.generate_reminder(30, logger, "Encoder",
                                                                     "Low Encoding speed detected, Please check your encode settings to avoid performance issues")
                if frame is not None:
                    frame_written = True
                    frame_writer.writeFrame(frame)
                self.reminder_bearer.terminate_reminder(reminder_id)

                chunk_frame_cnt += 1
                self.ARGS.task_info.update({"chunk_cnt": chunk_cnt, "render": now_frame})  # update render info

                if not chunk_frame_cnt % self.ARGS.render_gap:
                    frame_writer.close()
                    if concat_test_flag:
                        self.__check_audio_concat(chunk_tmp_path)
                        concat_test_flag = False
                    self.__rename_chunk(chunk_tmp_path, chunk_cnt, start_frame, now_frame)
                    chunk_cnt += 1
                    start_frame = now_frame + 1
                    frame_writer = self.__generate_frame_writer(start_frame, chunk_tmp_path, )
            if not self.ARGS.is_no_concat and not self.ARGS.is_img_output and not self._kill:
                self.concat_all()

        except Exception as e:
            logger.critical("Render Thread Panicked")
            logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
            return

        return

    def update_steam_flow(self, steam_flow: SteamFlow):
        self.steam_flow = steam_flow


class SuperResolutionFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, _reader_queue: Queue, _render_queue: Queue):
        super().__init__(_args)
        self.name = 'SuperResolution'
        self._input_queue = _reader_queue
        self._output_queue = _render_queue
        self.sr_module = SuperResolution()  # 超分类
        if not self.ARGS.use_sr:
            return
        sr_scale = self.ARGS.resize_exp  # TODO evict this later
        resize_param = self.ARGS.frame_size
        if all(self.ARGS.transfer_param):
            resize_param = self.ARGS.transfer_param
        if all(resize_param) and all(self.ARGS.resize_param):  # if frame_size == 0,0, sr_scale remains untouched
            sr_scale = (self.ARGS.resize_param[0] * self.ARGS.resize_param[1]) / (resize_param[0] * resize_param[1])
            sr_scale = int(math.ceil(math.sqrt(sr_scale)))
        if sr_scale == 1 and self.ARGS.resize_exp:
            sr_scale = self.ARGS.resize_exp
        try:
            if self.ARGS.use_sr_algo == "waifu2x":
                import Utils.SuperResolutionModule
                self.sr_module = Utils.SuperResolutionModule.SvfiWaifu(model=self.ARGS.use_sr_model,
                                                                       scale=sr_scale,
                                                                       num_threads=self.ARGS.ncnn_thread,
                                                                       resize=resize_param)
            elif self.ARGS.use_sr_algo == "realSR":
                import Utils.SuperResolutionModule
                self.sr_module = Utils.SuperResolutionModule.SvfiRealSR(model=self.ARGS.use_sr_model,
                                                                        scale=sr_scale,
                                                                        resize=resize_param)
            elif self.ARGS.use_sr_algo == "realESR":
                import Utils.RealESRModule
                self.sr_module = Utils.RealESRModule.SvfiRealESR(model=self.ARGS.use_sr_model,
                                                                 gpu_id=self.ARGS.use_specific_gpu,
                                                                 # TODO Assign another card here
                                                                 scale=sr_scale, tile=self.ARGS.sr_tilesize,
                                                                 half=self.ARGS.use_realesr_fp16,
                                                                 resize=resize_param)
            logger.info(
                f"Load SuperResolutionModule at {self.ARGS.use_sr_algo}, "
                f"Model at {self.ARGS.use_sr_model}, scale_times = {sr_scale}")
        except ImportError:
            logger.error(
                f"Import SR Module failed\n"
                f"{traceback.format_exc(limit=ArgumentManager.traceback_limit)}")
            self.ARGS.use_sr = False

    def run(self):
        """
        SR thread
        :return:
        """
        self._release_initiation()
        try:
            while True:
                if self._kill:
                    logger.warning("Super Resolution thread killed, break")
                    break
                task_acquire_time = time.time()
                task = self._input_queue.get()
                task_acquire_time = time.time() - task_acquire_time
                if task is None:
                    break
                if self.ARGS.use_sr:
                    """
                        task = {"now_frame", "img0", "img1", "n","scale", "is_end", "is_scene", "add_scene"}
                    """
                    now_frame = task["now_frame"]
                    img0 = task["img0"]
                    img1 = task["img1"]
                    reminder_id = self.reminder_bearer.generate_reminder(60, logger,
                                                                         "Super Resolution",
                                                                         "Low Super Resolution speed detected, Please consider lower your output settings to enhance speed")
                    sr_process_time = time.time()
                    if img0 is not None:
                        img0 = self.sr_module.svfi_process(img0)
                    sr_process_time = time.time() - sr_process_time
                    if img1 is not None:
                        img1 = self.sr_module.svfi_process(img1)
                    task['img0'] = img0
                    task['img1'] = img1
                    self.ARGS.update_task_info({'sr_now_frame': now_frame,
                                                'sr_task_acquire_time': task_acquire_time,
                                                'sr_process_time': sr_process_time,
                                                'sr_queue_len': self._input_queue.qsize()})
                    self.reminder_bearer.terminate_reminder(reminder_id)
                self._output_queue.put(task)
        except Exception as e:
            logger.critical("Super Resolution Thread Panicked")
            logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self._output_queue.put(None)
            self.ARGS.save_main_error(e)
            return
        self._task_done()
        return


class ProgressUpdateFlow(IOFlow):
    def __init__(self, _args: TaskArgumentManager, _read_flow: ReadFlow):
        super().__init__(_args)
        self.name = 'Pbar'
        self.start_frame = 0
        self.read_flow = _read_flow

    def update_start_frame(self, start_frame: int):
        self.start_frame = start_frame

    def run(self):
        """
        Start Progress Update Bar
        :return:
        """
        """(chunk_cnt, start_frame, end_frame, frame_cnt)"""
        self.read_flow.acquire_initiation_clock()
        pbar = tqdm.tqdm(total=self.ARGS.all_frames_cnt, unit="frames")
        pbar.moveto(n=self.start_frame)
        pbar.unpause()
        previous_cnt = self.start_frame
        self._release_initiation()
        while True:
            if self._kill:
                break
            task_status = self.ARGS.task_info  # render status quo
            if self.ARGS.render_only or self.ARGS.extract_only:
                now_frame = task_status['render']
                pbar_description = f"Process at Frame {now_frame}"
                postfix_dict = {"R": f"{task_status['render']}", "C": f"{now_frame}", }
            else:
                now_frame = task_status['rife_now_frame']
                pbar_description = f"Process at Chunk {task_status['chunk_cnt']:0>3d}"
                postfix_dict = {"R": f"{task_status['render']}",
                                "C": f"{now_frame}",
                                "S": f"{task_status['recent_scene']}",
                                "SC": f"{task_status['scene_cnt']}",
                                "RPT": f"{task_status['decode_process_time']:.2f}s",
                                "TAT": f"{task_status['rife_task_acquire_time']:.2f}s",
                                "PT": f"{task_status['rife_process_time']:.2f}s",
                                "QL": f"{task_status['rife_queue_len']}"}
            if self.ARGS.use_sr:
                postfix_dict.update({'SR': f"{task_status['sr_now_frame']}",
                                     'SRTAT': f"{task_status['sr_task_acquire_time']:.2f}s",
                                     'SRPT': f"{task_status['sr_process_time']:.2f}s",
                                     "SRL": f"{task_status['sr_queue_len']}", })

            pbar.set_description(pbar_description)
            pbar.set_postfix(postfix_dict)
            pbar.update(now_frame - previous_cnt)
            previous_cnt = now_frame
            time.sleep(0.2)
        pbar.update(abs(self.ARGS.all_frames_cnt - previous_cnt))
        pbar.close()


class InterpWorkFlow:
    # @profile
    def __init__(self, __args: TaskArgumentManager, **kwargs):
        global logger
        self.ARGS = __args
        self.run_all_time = datetime.datetime.now()

        """EULA"""
        self.eula = EULAWriter()
        self.eula.boom()

        """Set Queues"""
        queue_len = self.ARGS.frames_queue_len
        self.read_task_queue = Queue(maxsize=queue_len)
        self.sr_task_queue = Queue(maxsize=queue_len)
        self.rife_task_queue = Queue(maxsize=queue_len)
        self.render_task_queue = Queue(maxsize=queue_len)

        """Set Flow"""
        """self.steam_flow = SteamFlow(self.ARGS)
        self.sr_flow = SuperResolutionFlow(self.ARGS, self.read_task_queue, self.rife_task_queue)
        if not self.ARGS.use_sr:
            self.read_flow = ReadFlow(self.ARGS, self.rife_task_queue)
            self.sr_flow.kill()
        else:
            self.read_flow = ReadFlow(self.ARGS, self.read_task_queue)
        self.render_flow = RenderFlow(self.ARGS, self.render_task_queue)
        self.update_progress_flow = ProgressUpdateFlow(self.ARGS, self.read_flow)"""
        self.steam_flow = SteamFlow(self.ARGS)
        self.read_flow = ReadFlow(self.ARGS, self.read_task_queue)
        self.sr_flow = SuperResolutionFlow(self.ARGS, self.read_task_queue, self.rife_task_queue)
        self.render_flow = RenderFlow(self.ARGS, self.render_task_queue)
        self.update_progress_flow = ProgressUpdateFlow(self.ARGS, self.read_flow)

        """Set VFI Core"""
        self.vfi_core = VideoFrameInterpolation(self.ARGS)

        """Set 'Global' Reminder"""
        self.reminder_bearer = OverTimeReminderBearer()

    def feed_to_render(self, frames_list: list, is_end=False):
        """
        维护输出帧数组的输入（往输出渲染线程喂帧
        :param frames_list:
        :param is_end: 是否是视频结尾
        :return:
        """
        for frame_i, frame_data in enumerate(frames_list):
            if frame_data is None:
                self.render_task_queue.put(None)
                logger.debug("Put None to render_task_queue in advance")
                return
            self.render_task_queue.put(frame_data)  # 往输出队列（消费者）喂正常的帧
            if frame_i == len(frames_list) - 1:
                if is_end:
                    self.render_task_queue.put(None)
                    logger.debug("Put None to render_task_queue")
                    return
        pass

    def nvidia_vram_test(self):
        """
        显存测试
        :return:
        """
        try:
            if self.ARGS.resize_width and self.ARGS.resize_height:
                w, h = self.ARGS.resize_width, self.ARGS.resize_height
            else:
                w, h = self.ARGS.video_info_instance.frames_size

            logger.info(f"Start VRAM Test: {w}x{h} with scale {self.ARGS.rife_scale}")

            test_img0, test_img1 = np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8), \
                                   np.random.randint(0, 255, size=(w, h, 3)).astype(np.uint8)
            self.vfi_core.generate_n_interp(test_img0, test_img1, 1, self.ARGS.rife_scale)
            logger.info(f"VRAM Test Success, Resume of workflow ahead")
            del test_img0, test_img1
        except Exception as e:
            logger.error("VRAM Check Failed, PLS Lower your presets\n" + traceback.format_exc(
                limit=ArgumentManager.traceback_limit))
            raise e

    def check_params_with_steam(self):
        if self.ARGS.is_steam:
            if not self.steam_flow.steam_valid:
                error = str(self.steam_flow.steam_error).split('\n')[-1]
                logger.error(f"Steam Validation Failed: {error}")
                return
            else:
                valid_response = self.steam_flow.CheckSteamAuth()
                if valid_response != 0:
                    logger.error(f"Steam Validation Failed, code {valid_response}")
                    raise GenericSteamException(f"Steam Validation Failed, code {valid_response}")

            steam_dlc_check = self.steam_flow.CheckProDLC(0)
            if not steam_dlc_check:
                _msg = "SVFI - Professional DLC Not Purchased,"
                if self.ARGS.extract_only or self.ARGS.render_only:
                    raise GenericSteamException(f"{_msg} Extract/Render ToolBox Unavailable")
                if self.ARGS.input_start_point is not None or self.ARGS.input_end_point is not None:
                    raise GenericSteamException(f"{_msg} Manual Input Section Unavailable")
                if self.ARGS.is_scdet_output or self.ARGS.is_scdet_mix:
                    raise GenericSteamException(f"{_msg} Scdet Output/Mix Unavailable")
                if self.ARGS.use_sr:
                    raise GenericSteamException(f"{_msg} Super Resolution Module Unavailable")
                if self.ARGS.use_rife_multi_cards:
                    raise GenericSteamException(f"{_msg} Multi Video Cards Work flow Unavailable")

    def check_interp_prerequisite(self):
        if self.ARGS.render_only or self.ARGS.extract_only or self.ARGS.concat_only:
            return
        if self.ARGS.use_ncnn:
            self.ARGS.rife_model_name = os.path.basename(self.ARGS.rife_model)
            from Utils import inference_rife_ncnn as inference
        else:
            try:
                # raise Exception("Load Torch Failed Test")
                from Utils import inference_rife as inference
            except Exception:
                logger.warning("Import Torch Failed, use NCNN-RIFE instead")
                logger.error(traceback.format_exc(limit=ArgumentManager.traceback_limit))
                self.ARGS.use_ncnn = True
                self.ARGS.rife_model = "rife-v2"
                self.ARGS.rife_model_name = "rife-v2"
                from Utils import inference_rife_ncnn as inference
        """Update RIFE Core"""
        self.vfi_core = inference.RifeInterpolation(self.ARGS)
        self.vfi_core.initiate_algorithm(self.ARGS)

        if not self.ARGS.use_ncnn:
            self.nvidia_vram_test()
        self.read_flow.update_vfi_core(self.vfi_core)

    def check_outside_error(self):
        if self.ARGS.get_main_error() is not None:
            logger.error("Error outside RIFE:")
            self.feed_to_render([None], is_end=True)
            raise self.ARGS.get_main_error()

    def task_finish(self):
        if self.ARGS.get_main_error() is not None:
            logger.error("Error after finish RIFE Task:")
            self.feed_to_render([None], is_end=True)
            raise self.ARGS.get_main_error()
        logger.info(f"Program finished at {datetime.datetime.now()}: "
                    f"Duration: {datetime.datetime.now() - self.run_all_time}")
        logger.info("Please Note That Commercial Use of SVFI's Output is Strictly PROHIBITED, "
                    "Check EULA for more details")
        self.reminder_bearer.terminate_all()
        utils_overtime_reminder_bearer.terminate_all()
        pass

    def task_failed(self):
        self.read_flow.kill()
        self.render_flow.kill()
        self.sr_flow.kill()
        self.update_progress_flow.kill()
        self.reminder_bearer.terminate_all()
        utils_overtime_reminder_bearer.terminate_all()
        logger.info(f"\n\n\nProgram Failed at {datetime.datetime.now()}: "
                    f"Duration: {datetime.datetime.now() - self.run_all_time}")
        if self.ARGS.get_main_error() is not None:
            raise self.ARGS.get_main_error()

    def run(self):
        """
        Main Thread of SVFI
        :return:
        """

        """Check Steam Validation"""
        self.check_params_with_steam()

        """Go through the process"""
        if self.ARGS.concat_only:
            self.render_flow.concat_all()
            self.task_finish()
            return
        """Concat Already / Mission Conflict Check & Dolby Vision Sort"""
        if self.render_flow.check_concat_result():
            self.task_finish()
            return

        """Load RIFE Model"""
        self.check_interp_prerequisite()

        """Get RIFE Task Thread"""
        self.read_flow.start()
        self.render_flow.start()
        self.sr_flow.start()
        self.update_progress_flow.start()

        PURE_SCENE_THRESHOLD = 30

        self.check_outside_error()
        self.read_flow.acquire_initiation_clock()
        self.render_flow.acquire_initiation_clock()
        self.update_progress_flow.acquire_initiation_clock()

        try:
            while True:
                task_acquire_time = time.time()
                task = self.rife_task_queue.get(timeout=3600)
                task_acquire_time = time.time() - task_acquire_time
                process_time = time.time()
                if task is None:
                    self.feed_to_render([None], is_end=True)
                    break
                """
                task = {"now_frame", "img0", "img1", "n","scale", "is_end", "is_scene", "add_scene"}
                """
                now_frame = task["now_frame"]
                img0 = task["img0"]
                img1 = task["img1"]
                n = task["n"]
                scale = task["scale"]
                is_end = task["is_end"]
                add_scene = task["add_scene"]

                debug = False
                """Test
                1. 正常4K，解码编码
                2. 一拍N卡顿
                """

                if img1 is None:
                    self.feed_to_render([None], is_end=True)
                    break

                if all(self.ARGS.resize_param):
                    img0 = cv2.resize(img0, (self.ARGS.resize_param[0], self.ARGS.resize_param[1]),
                                      interpolation=cv2.INTER_LANCZOS4)
                    img1 = cv2.resize(img1, (self.ARGS.resize_param[0], self.ARGS.resize_param[1]),
                                      interpolation=cv2.INTER_LANCZOS4)

                frames_list = [img0]
                if self.ARGS.is_scdet_mix and add_scene:
                    mix_list = Tools.get_mixed_scenes(img0, img1, n + 1)
                    frames_list.extend(mix_list)
                else:
                    reminder_id = self.reminder_bearer.generate_reminder(60, logger,
                                                                         "Video Frame Interpolation",
                                                                         "Low interpolate speed detected, Please consider lower your output settings to enhance speed")
                    if n > 0:
                        if n > PURE_SCENE_THRESHOLD and Tools.check_pure_img(img0):
                            """It's Pure Img Sequence, Copy img0"""
                            for i in range(n):
                                frames_list.append(img0)
                        else:
                            interp_list = self.vfi_core.generate_n_interp(img0, img1, n=n, scale=scale, debug=debug)
                            frames_list.extend(interp_list)
                    if add_scene:  # [AA BBB CC DDD] E
                        frames_list.append(img1)
                    self.reminder_bearer.terminate_reminder(reminder_id)

                feed_list = list()
                for i in frames_list:
                    feed_list.append([now_frame, i])
                if self.ARGS.use_evict_flicker or self.ARGS.use_rife_fp16:
                    img_ori = frames_list[0].copy()
                    frames_list[0] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                     debug=debug)
                    if add_scene:
                        img_ori = frames_list[-1].copy()
                        frames_list[-1] = self.vfi_core.generate_n_interp(img_ori, img_ori, n=1, scale=scale,
                                                                          debug=debug)

                process_time = time.time() - process_time
                self.update_rife_progress(now_frame, task_acquire_time, process_time)
                self.feed_to_render(feed_list, is_end=is_end)
                if is_end:
                    break
        except Exception as e:
            logger.critical("Main Thread Panicked")
            logger.critical(traceback.format_exc(limit=ArgumentManager.traceback_limit))
            self.ARGS.save_main_error(e)
            self.task_failed()
            return
        if self.ARGS.get_main_error() is not None:
            """Shit happened after receiving None as end signal"""
            self.task_failed()
            return
        while self.render_flow.is_alive() or self.sr_flow.is_alive() or self.read_flow.is_alive():
            """等待渲染线程结束"""
            time.sleep(0.1)
        self.update_progress_flow.kill()
        self.task_finish()

    def update_rife_progress(self, now_frame, task_acquire_time, process_time, ):
        update_dict = {'rife_now_frame': now_frame,
                       'rife_task_acquire_time': task_acquire_time,
                       'rife_process_time': process_time,
                       'rife_queue_len': self.rife_task_queue.qsize()}
        self.ARGS.update_task_info(update_dict)
        pass


GLOBAL_ARGS = TaskArgumentManager(global_args)

"""设置可见的gpu"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if int(GLOBAL_ARGS.rife_cuda_cnt) != 0 and GLOBAL_ARGS.use_rife_multi_cards:
    cuda_devices = [str(i) for i in range(GLOBAL_ARGS.rife_cuda_cnt)]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(cuda_devices)}"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GLOBAL_ARGS.use_specific_gpu}"

"""强制使用CPU"""
if GLOBAL_ARGS.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f""

global_workflow = InterpWorkFlow(GLOBAL_ARGS)
global_workflow.run()
sys.exit(0)
