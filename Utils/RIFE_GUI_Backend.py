import datetime
import hashlib
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import subprocess as sp
import sys
import time
import traceback

import cv2
import psutil
import torch
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from Utils import SVFI_UI, SVFI_help, SVFI_about, SVFI_preference
from Utils.utils import Utils, EncodePresetAssemply

MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False

Utils = Utils()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)

appDataPath = os.path.join(dname, "SVFI.ini")
appData = QSettings(appDataPath, QSettings.IniFormat)
appData.setIniCodec("UTF-8")

logger = Utils.get_logger("GUI", dname)
ols_potential = os.path.join(dname, "one_line_shot_args.exe")
ico_path = os.path.join(dname, "svfi.ico")


class SVFI_Config_Manager:
    """
    SVFI 配置文件管理类
    """

    def __init__(self, **kwargs):
        self.filename = ""
        self.dirname = dname
        self.SVFI_config_path = os.path.join(self.dirname, "SVFI.ini")
        pass

    def FetchConfig(self, filename):
        """
        根据输入文件名获得配置文件路径
        :param filename:
        :return:
        """
        config_path = self.__generate_config_path(filename)
        if os.path.exists(config_path):
            return config_path
        else:
            return None

    def DuplicateConfig(self, filename):
        """
        复制配置文件
        :param filename:
        :return:
        """
        config_path = self.__generate_config_path(filename)
        if not os.path.exists(self.SVFI_config_path):
            raise OSError("Not find Config")
        shutil.copy(self.SVFI_config_path, config_path)
        pass

    def RemoveConfig(self, filename):
        """
        移除配置文件
        :param filename:
        :return:
        """
        config_path = self.__generate_config_path(filename)
        if os.path.exists(config_path):
            os.remove(config_path)
        else:
            logger.warning("Not find Config to remove, guess executed directly from main file")
        pass

    def MaintainConfig(self, filename):
        """
        维护配置文件
        :param filename:
        :return:
        """
        config_path = self.__generate_config_path(filename)
        if os.path.exists(config_path):
            shutil.copy(config_path, self.SVFI_config_path)
            return True
        else:
            return False
        pass

    def __generate_config_path(self, filename):
        m = hashlib.md5(filename.encode(encoding='utf-8'))
        return os.path.join(self.dirname, f"SVFI_Config_{m.hexdigest()[:6]}.ini")


class SVFI_Help_Dialog(QDialog, SVFI_help.Ui_Dialog):
    def __init__(self, parent=None):
        super(SVFI_Help_Dialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)


class SVFI_About_Dialog(QDialog, SVFI_about.Ui_Dialog):
    def __init__(self, parent=None):
        super(SVFI_About_Dialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)


class SVFI_Preference_Dialog(QDialog, SVFI_preference.Ui_Dialog):
    preference_signal = pyqtSignal(dict)

    def __init__(self, parent=None, preference_dict=None):
        super(SVFI_Preference_Dialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        self.preference_dict = preference_dict
        self.update_preference()
        self.ExpertModeChecker.clicked.connect(self.request_preference)
        self.buttonBox.clicked.connect(self.request_preference)

    def closeEvent(self, event):
        self.request_preference()

    def close(self):
        self.request_preference()

    def update_preference(self):
        """
        初始化，更新偏好设置
        :return:
        """
        if self.preference_dict is None:
            return
        assert type(self.preference_dict) == dict

        self.MultiTaskRestChecker.setChecked(self.preference_dict["multi_task_rest"])
        self.MultiTaskRestInterval.setValue(self.preference_dict["multi_task_rest_interval"])

        self.AfterMission.setCurrentIndex(self.preference_dict["after_mission"])  # None

        self.ForceCpuChecker.setChecked(self.preference_dict["force_cpu"])

        self.ExpertModeChecker.setChecked(self.preference_dict["expert"])
        pass

    def request_preference(self):
        """
        申请偏好设置更改
        :return:
        """
        preference_dict = dict()
        preference_dict["multi_task_rest"] = self.MultiTaskRestChecker.isChecked()
        preference_dict["multi_task_rest_interval"] = self.MultiTaskRestInterval.value()
        preference_dict["after_mission"] = self.AfterMission.currentIndex()
        preference_dict["force_cpu"] = self.ForceCpuChecker.isChecked()
        preference_dict["expert"] = self.ExpertModeChecker.isChecked()
        self.preference_signal.emit(preference_dict)


class SVFI_Run_Others(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, command, task_id=0, data=None, parent=None):
        """
        多线程运行系统命令
        :param command:
        :param task_id:
        :param data: 信息回传时的数据
        :param parent:
        """
        super(SVFI_Run_Others, self).__init__(parent)
        self.command = command
        self.task_id = task_id
        self.data = data

    def fire_finish_signal(self):
        emit_json = {"id": self.task_id, "status": 1, "data": self.data}
        self.run_signal.emit(json.dumps(emit_json))

    def run(self):
        logger.info(f"[CMD Thread]: Start execute {self.command}")
        ps = subprocess.Popen(self.command)
        ps.wait()
        self.fire_finish_signal()
        pass

    pass


class SVFI_Run(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, parent=None, concat_only=False, extract_only=False, render_only=False):
        """
        
        :param parent:
        :param concat_only:
        :param extract_only:
        """
        super(SVFI_Run, self).__init__(parent)
        self.concat_only = concat_only
        self.extract_only = extract_only
        self.render_only = render_only
        self.command = ""
        self.current_proc = None
        self.kill = False
        self.pause = False
        self.task_cnt = 0
        self.silent = False
        self.config_maintainer = SVFI_Config_Manager()
        self.current_filename = ""
        self.current_step = 0

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self, input_file):
        if os.path.splitext(appData.value("OneLineShotPath"))[-1] == ".exe":
            self.command = appData.value("OneLineShotPath") + " "
        else:
            self.command = f'python "{appData.value("OneLineShotPath")}" '

        if not len(input_file) or not os.path.exists(input_file):
            self.command = ""
            return ""

        if float(appData.value("fps", -1.0, type=float)) <= 0 or float(
                appData.value("target_fps", -1.0, type=float)) <= 0:
            logger.error("Find Invalid FPS/Target_FPS")
            return ""

        self.command += f'--input {Utils.fillQuotation(input_file)} '

        output_path = appData.value("output")
        if os.path.isfile(output_path):
            logger.info("OutputPath with FileName detected")
            appData.setValue("output", os.path.dirname(output_path))
        self.command += f'--output {Utils.fillQuotation(output_path)} '

        config_path = self.config_maintainer.FetchConfig(input_file)
        if config_path is None:
            self.command += f'--config {Utils.fillQuotation(appDataPath)} '
        else:
            self.command += f'--config {Utils.fillQuotation(config_path)} '

        """Alternative Mission Field"""
        if self.concat_only:
            self.command += f"--concat-only "
        if self.extract_only:
            self.command += f"--extract-only "
        if self.render_only:
            self.command += f"--render-only "

        self.command = self.command.replace("\\", "/")
        return self.command

    def update_status(self, finished=False, notice="", sp_status="", returncode=-1):
        """
        update sub process status
        :return:
        """
        emit_json = {"cnt": self.task_cnt, "current": self.current_step, "finished": finished,
                     "notice": notice, "subprocess": sp_status, "returncode": returncode}
        emit_json = json.dumps(emit_json)
        self.run_signal.emit(emit_json)

    def maintain_multitask(self):
        appData.setValue("chunk", 1)
        appData.setValue("interp_start", 0)
        appData.setValue("start_point", "00:00:00")
        appData.setValue("end_point", "00:00:00")

    def run(self):
        logger.info("SVFI Task Run")
        file_list = appData.value("InputFileName", "").split(";")

        command_list = list()
        for f in file_list:
            command = self.build_command(f)
            if not len(command):
                continue
            command_list.append((f, command))

        self.current_step = 0
        self.task_cnt = len(command_list)

        if self.task_cnt > 1:
            """MultiTask"""
            # appData.setValue("output_only", True)
            appData.setValue("batch", True)

        if not self.task_cnt:
            logger.info("Task List Empty, Please Check Your Settings! (input fps for example)")
            self.update_status(True, "\nTask List is Empty!\n")
            return

        interval_time = time.time()
        try:
            for f in command_list:
                logger.info(f"Designed Command:\n{f}")
                proc_args = shlex.split(f[1])
                self.current_proc = sp.Popen(args=proc_args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='gb18030',
                                             # TODO Check UTF-8
                                             errors='ignore',
                                             universal_newlines=True)

                flush_lines = ""
                while self.current_proc.poll() is None:
                    if self.kill:
                        break

                    if self.pause:
                        pid = self.current_proc.pid
                        pause = psutil.Process(pid)  # 传入子进程的pid
                        pause.suspend()  # 暂停子进程
                        self.update_status(False, notice=f"\n\nWARNING, 补帧已被手动暂停", returncode=-1)

                        while True:
                            if self.kill:
                                break
                            elif not self.pause:
                                pause.resume()
                                self.update_status(False, notice=f"\n\nWARNING, 补帧已继续",
                                                   returncode=-1)
                                break
                            time.sleep(0.2)
                    else:
                        line = self.current_proc.stdout.readline()
                        self.current_proc.stdout.flush()

                        """Replace Field"""
                        flush_lines += line.replace("[A", "")

                        if "error" in flush_lines.lower():
                            """Imediately Upload"""
                            logger.error(f"[In ONE LINE SHOT]: f{flush_lines}")
                            self.update_status(False, sp_status=f"{flush_lines}")
                            flush_lines = ""
                        elif len(flush_lines) and time.time() - interval_time > 0.1:
                            interval_time = time.time()
                            self.update_status(False, sp_status=f"{flush_lines}")
                            flush_lines = ""

                self.update_status(False, sp_status=f"{flush_lines}")  # emit last possible infos

                self.current_step += 1
                self.update_status(False, f"\nINFO - {datetime.datetime.now()} {f[0]} 完成\n\n")
                self.maintain_multitask()
                self.config_maintainer.RemoveConfig(f[0])

        except Exception:
            logger.error(traceback.format_exc())

        self.update_status(True, returncode=self.current_proc.returncode)
        logger.info("Tasks Finished")
        if self.current_proc.returncode == 0:
            """Finish Normally"""
            if appData.value("after_mission", type=int) == 1:
                logger.info("Task Finished Normally, User Request to Shutdown")
                os.system("shutdown -s -t 120")  # TODO Debug After Mission
            elif appData.value("after_mission", type=int) == 2:
                logger.info("Task Finished Normally, User Request to Hibernate")
                os.system("shutdown -h")

        pass

    def kill_proc_exec(self):
        self.kill = True
        if self.current_proc is not None:
            self.current_proc.terminate()
            self.update_status(False, notice=f"\n\nWARNING, 补帧已被强制结束", returncode=-1)
            logger.info("Kill Process")
        else:
            logger.warning("There's no Process to kill")

    def pause_proc_exec(self):
        self.pause = not self.pause
        if self.pause:
            logger.info("Pause Process Command Fired")
        else:
            logger.info("Resume Process Command Fired")

    pass


class RoundShadow(QWidget):
    def __init__(self, parent=None):
        """圆角边框类"""
        super(RoundShadow, self).__init__(parent)
        self.border_width = 8
        # 设置 窗口无边框和背景透明 *必须
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

    def paintEvent(self, event):
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)

        pat = QPainter(self)
        pat.setRenderHint(pat.Antialiasing)
        pat.fillPath(path, QBrush(Qt.white))

        color = QColor(192, 192, 192, 50)

        for i in range(10):
            i_path = QPainterPath()
            i_path.setFillRule(Qt.WindingFill)
            ref = QRectF(10 - i, 10 - i, self.width() - (10 - i) * 2, self.height() - (10 - i) * 2)
            # i_path.addRect(ref)
            i_path.addRoundedRect(ref, self.border_width, self.border_width)
            color.setAlpha(150 - i ** 0.5 * 50)
            pat.setPen(color)
            pat.drawPath(i_path)

        # 圆角
        pat2 = QPainter(self)
        pat2.setRenderHint(pat2.Antialiasing)  # 抗锯齿
        pat2.setBrush(Qt.white)
        pat2.setPen(Qt.transparent)

        rect = self.rect()
        rect.setLeft(9)
        rect.setTop(9)
        rect.setWidth(rect.width() - 9)
        rect.setHeight(rect.height() - 9)
        pat2.drawRoundedRect(rect, 4, 4)


class RIFE_GUI_BACKEND(QMainWindow, SVFI_UI.Ui_MainWindow):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None, free=False, version="0.0.0 beta"):
        """
        SVFI 主界面类初始化方法
        添加新选项/变量 3/3 实现先于load_current_settings的特殊新配置
        :param parent:
        :param free:
        """
        super(RIFE_GUI_BACKEND, self).__init__()
        self.setupUi(self)
        self.thread = None
        self.Exp = int(math.log(float(appData.value("exp", "2")), 2))
        self.version = version
        self.free = free
        if self.free:
            self.settings_free_hide()

        appData.setValue("app_path", ddname)

        if appData.value("ffmpeg", "") != "ffmpeg":
            self.ffmpeg = f'"{os.path.join(appData.value("ffmpeg", ""), "ffmpeg.exe")}"'
        else:
            self.ffmpeg = appData.value("ffmpeg", "")

        # debug
        # self.ffmpeg = '"D:/60-fps-Project/Projects/RIFE GUI/release/SVFI.Env/神威 SVFI 3.2/Package/ffmpeg.exe"'

        if os.path.exists(appDataPath):
            logger.info("Previous Settings Found")

        self.check_gpu = False
        self.silent = False
        self.tqdm_re = re.compile(".*?Process at .*?\]")
        self.current_failed = False
        self.formatted_option_check = []
        self.pause = False

        """Preference Maintainer"""
        self.multi_task_rest = False
        self.multi_task_rest_interval = 0
        self.after_mission = 0
        self.force_cpu = False
        self.expert_mode = True
        self.SVFI_Preference_form = None

        """Initiate and Check GPU"""
        self.settings_initiation()
        self.hasNVIDIA = True
        self.settings_update_gpu_info()
        self.settings_update_model_info()
        self.on_HwaccelSelector_currentTextChanged()  # Flush Encoder Sets
        self.on_ExpertMode_changed()
        self.on_UseAiSR_clicked()
        self.on_UseEncodeThread_clicked()
        self.on_slowmotion_clicked()
        self.on_MBufferChecker_clicked()
        self.on_DupRmMode_currentTextChanged()
        self.on_ScedetChecker_clicked()
        self.on_EncoderSelector_currentTextChanged()
        self.settings_initiation()  # A double initiation to save encoding settings

        """Initiate Beautiful Layout and Signals"""
        self.AdvanceSettingsArea.setVisible(False)
        self.ProgressBarVisibleControl.setVisible(False)

        """Link InputFileName Event"""
        self.InputFileName.clicked.connect(self.on_InputFileName_currentItemChanged)
        self.InputFileName.itemClicked.connect(self.on_InputFileName_currentItemChanged)
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)

        """Table Maintainer"""
        # self.
        """Dilapidation Maintainer"""
        self.settings_dilapidation_hide()

    def settings_dilapidation_hide(self):
        """Hide Dilapidated Options"""
        self.TtaModeChecker.setVisible(False)
        self.TtaModeChecker.setChecked(False)
        self.ScdetModeLabel.setVisible(False)
        self.ScdetMode.setVisible(False)
        self.ScdetFlowLen.setVisible(False)
        self.AutoInterpScaleChecker.setVisible(False)

    def settings_free_hide(self):
        """
        ST only
        :return:
        """
        # self.DupRmChecker.setVisible(False)
        self.DupFramesTSelector.setVisible(False)
        self.DupFramesTSelector.setValue(0.2)
        self.DupRmMode.clear()
        ST_RmMode = ["不去除重复帧", "单一识别"]
        for m in ST_RmMode:
            self.DupRmMode.addItem(m)

        self.StartPoint.setVisible(False)
        self.EndPoint.setVisible(False)
        self.StartPointLabel.setVisible(False)
        self.EndPointLabel.setVisible(False)
        self.ScdetUseMix.setVisible(False)
        self.UseAiSR.setVisible(False)
        self.RenderOnlyGroupbox.setVisible(False)

    def settings_initiation(self):
        """
        初始化用户选项载入
        从配置文件中读取上一次设置并初始化页面
        添加新选项/变量 1/3 appData -> Options
        :return:
        """
        input_list = appData.value("InputFileName", "").split(";")
        if not len(self.function_get_input_files()):
            for i in input_list:
                if len(i):
                    self.InputFileName.addItem(i)

        appData.setValue("OneLineShotPath", ols_potential)
        appData.setValue("ffmpeg", dname)

        if not os.path.exists(ols_potential):
            appData.setValue("OneLineShotPath",
                             r"D:\60-fps-Project\Projects\RIFE GUI\one_line_shot_args.py")
            appData.setValue("ffmpeg", "ffmpeg")
            logger.info("Change to Debug Path")

        """Basic Configuration"""
        self.OutputFolder.setText(appData.value("output"))
        self.InputFPS.setText(appData.value("fps", "0"))
        self.OutputFPS.setText(appData.value("target_fps"))
        self.ExpSelecter.setCurrentText("x" + str(2 ** int(appData.value("exp", "1"))))
        self.ImgOutputChecker.setChecked(appData.value("img_output", False, type=bool))
        appData.setValue("img_input", appData.value("img_input", False))
        appData.setValue("batch", False)
        self.KeepChunksChecker.setChecked(not appData.value("output_only", True, type=bool))
        self.StartPoint.setTime(QTime.fromString(appData.value("start_point", "00:00:00"), "HH:mm:ss"))
        self.EndPoint.setTime(QTime.fromString(appData.value("end_point", "00:00:00"), "HH:mm:ss"))
        self.DebugChecker.setChecked(appData.value("debug", False, type=bool))

        """Output Resize Configuration"""
        self.CropHeightSettings.setValue(appData.value("crop_height", 0, type=int))
        self.CropWidthpSettings.setValue(appData.value("crop_width", 0, type=int))
        self.ResizeHeightSettings.setValue(appData.value("resize_height", 0, type=int))
        self.ResizeWidthSettings.setValue(appData.value("resize_width", 0, type=int))

        """Render Configuration"""
        self.UseCRF.setChecked(appData.value("use_crf", True, type=bool))
        self.CRFSelector.setValue(appData.value("crf", 16, type=int))
        self.UseTargetBitrate.setChecked(appData.value("use_bitrate", False, type=bool))
        self.BitrateSelector.setValue(appData.value("bitrate", 90, type=float))
        # self.PresetSelector.setCurrentText(appData.value("preset", "slow"))
        self.HwaccelSelector.setCurrentText(appData.value("hwaccel_mode", "CPU", type=str))
        self.HwaccelPresetSelector.setCurrentText(appData.value("hwaccel_preset", "None"))
        self.HwaccelDecode.setChecked(appData.value("hwaccel_decode", True, type=bool))
        self.UseEncodeThread.setChecked(appData.value("use_encode_thread", False, type=bool))
        self.EncodeThreadSelector.setValue(appData.value("encode_thread", 16, type=int))
        self.EncoderSelector.setCurrentText(appData.value("encoder", "~"))
        self.FFmpegCustomer.setText(appData.value("ffmpeg_customized", ""))
        self.ExtSelector.setCurrentText(appData.value("output_ext", "mp4"))
        self.RenderGapSelector.setValue(appData.value("render_gap", 1000, type=int))
        self.SaveAudioChecker.setChecked(appData.value("save_audio", True, type=bool))
        self.FastDenoiseChecker.setChecked(appData.value("fast_denoise", False, type=bool))
        self.StrictModeChecker.setChecked(appData.value("strict_mode", False, type=bool))
        self.QuickExtractChecker.setChecked(appData.value("quick_extract", True, type=bool))
        self.DeinterlaceChecker.setChecked(appData.value("deinterlace", False, type=bool))

        """Slowmotion Configuration"""
        self.slowmotion.setChecked(appData.value("slow_motion", False, type=bool))
        self.SlowmotionFPS.setText(appData.value("slow_motion_fps", "", type=str))
        self.GifLoopChecker.setChecked(appData.value("gif_loop", True, type=bool))

        """Scdet and RD Configuration"""
        self.ScedetChecker.setChecked(not appData.value("no_scdet", False, type=bool))
        self.ScdetSelector.setValue(appData.value("scdet_threshold", 12, type=int))
        self.ScdetUseMix.setChecked(appData.value("scdet_mix", False, type=bool))
        self.ScdetOutput.setChecked(appData.value("scdet_output", False, type=bool))
        self.ScdetFlowLen.setCurrentIndex(appData.value("scdet_flow", 0, type=int))
        self.UseFixedScdet.setChecked(appData.value("use_fixed_scdet", False, type=bool))
        self.ScdetMaxDiffSelector.setValue(appData.value("fixed_max_scdet", 40, type=int))
        self.ScdetMode.setCurrentIndex(appData.value("scdet_mode", 0, type=int))
        # self.DupRmChecker.setChecked(appData.value("remove_dup", False, type=bool))
        self.DupRmMode.setCurrentIndex(appData.value("remove_dup_mode", 0, type=int))
        self.DupFramesTSelector.setValue(appData.value("dup_threshold", 10.00, type=float))

        """AI Super Resolution Configuration"""
        self.UseAiSR.setChecked(appData.value("use_sr", False, type=bool))
        self.on_UseAiSR_clicked()

        """RIFE Configuration"""
        self.FP16Checker.setChecked(appData.value("fp16", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("scale", "1.00"))
        self.ReverseChecker.setChecked(appData.value("reverse", False, type=bool))
        self.ForwardEnsembleChecker.setChecked(appData.value("forward_ensemble", False, type=bool))
        self.AutoInterpScaleChecker.setChecked(appData.value("auto_scale", False, type=bool))
        self.on_AutoInterpScaleChecker_clicked()
        self.UseNCNNButton.setChecked(appData.value("ncnn", False, type=bool))
        self.TtaModeChecker.setChecked(appData.value("tta_mode", False, type=bool))
        self.ncnnInterpThreadCnt.setValue(appData.value("ncnn_thread", 4, type=int))
        self.ncnnSelectGPU.setValue(appData.value("ncnn_gpu", 0, type=int))
        # Update RIFE Model
        rife_model_list = []
        for i in range(self.ModuleSelector.count()):
            rife_model_list.append(self.ModuleSelector.itemText(i))
        if appData.value("selected_model_name", "") in rife_model_list:
            self.ModuleSelector.setCurrentText(appData.value("selected_model_name", ""))

        """Multi Task Configuration"""
        self.multi_task_rest = appData.value("multi_task_rest", False, type=bool)
        self.multi_task_rest_interval = appData.value("multi_task_rest_interval", False, type=bool)
        self.after_mission = appData.value("after_mission", 0, type=int)
        self.force_cpu = appData.value("force_cpu", False, type=bool)
        self.expert_mode = appData.value("expert_mode", True, type=bool)

        """REM Management Configuration"""
        self.MBufferChecker.setChecked(appData.value("use_manual_buffer", False, type=bool))
        self.BufferSizeSelector.setValue(appData.value("manual_buffer_size", 1, type=int))

        """Maintain SVFI Startup Resolution"""
        desktop = QApplication.desktop()
        pos = appData.value("pos", QVariant(QPoint(960, 540)))
        size = appData.value("size", QVariant(QSize(int(desktop.width() * 0.6), int(desktop.height() * 0.5))))
        self.resize(size)
        self.move(pos)

        # self.setAttribute(Qt.WA_TranslucentBackground)

    def settings_load_current(self):
        """
        将现有界面的选项缓存至配置文件中
        添加新选项/变量 2/3 Options -> appData
        :return:
        """
        input_file_names = ""
        for i in self.function_get_input_files():
            if len(i):
                input_file_names += f"{i};"
        """Input Basic Input Information"""
        appData.setValue("version", self.version)
        appData.setValue("InputFileName", input_file_names)
        appData.setValue("output", self.OutputFolder.text())
        appData.setValue("fps", self.InputFPS.text())
        appData.setValue("target_fps", self.OutputFPS.text())
        appData.setValue("exp", int(math.log(int(self.ExpSelecter.currentText()[1:]), 2)))
        appData.setValue("img_output", self.ImgOutputChecker.isChecked())
        appData.setValue("output_only", not self.KeepChunksChecker.isChecked())
        appData.setValue("save_audio", self.SaveAudioChecker.isChecked())
        appData.setValue("output_ext", self.ExtSelector.currentText())

        """Input Time Stamp"""
        appData.setValue("start_point", self.StartPoint.time().toString("HH:mm:ss"))
        appData.setValue("end_point", self.EndPoint.time().toString("HH:mm:ss"))
        appData.setValue("chunk", self.StartChunk.value())
        appData.setValue("interp_start", self.StartFrame.value())
        appData.setValue("render_gap", self.RenderGapSelector.value())

        """Render"""
        appData.setValue("use_crf", self.UseCRF.isChecked())
        appData.setValue("use_bitrate", self.UseTargetBitrate.isChecked())
        appData.setValue("crf", self.CRFSelector.value())
        appData.setValue("bitrate", self.BitrateSelector.value())
        appData.setValue("preset", self.PresetSelector.currentText())
        appData.setValue("encoder", self.EncoderSelector.currentText())
        appData.setValue("hwaccel_mode", self.HwaccelSelector.currentText())
        appData.setValue("hwaccel_preset", self.HwaccelPresetSelector.currentText())
        appData.setValue("hwaccel_decode", self.HwaccelDecode.isChecked())
        appData.setValue("use_encode_thread", self.UseEncodeThread.isChecked())
        appData.setValue("encode_thread", self.EncodeThreadSelector.value())
        appData.setValue("quick_extract", self.QuickExtractChecker.isChecked())
        appData.setValue("strict_mode", self.StrictModeChecker.isChecked())
        appData.setValue("ffmpeg_customized", self.FFmpegCustomer.text())
        appData.setValue("no_concat", False)  # always concat
        appData.setValue("fast_denoise", self.FastDenoiseChecker.isChecked())

        """Special Render Effect"""
        appData.setValue("gif_loop", self.GifLoopChecker.isChecked())
        appData.setValue("slow_motion", self.slowmotion.isChecked())
        appData.setValue("slow_motion_fps", self.SlowmotionFPS.text())
        if appData.value("slow_motion", False, type=bool):
            appData.setValue("save_audio", False)
            self.SaveAudioChecker.setChecked(False)
        appData.setValue("deinterlace", self.DeinterlaceChecker.isChecked())

        height, width = self.ResizeHeightSettings.value(), self.ResizeWidthSettings.value()
        appData.setValue("resize_width", width)
        appData.setValue("resize_height", height)
        if all((width, height)):
            appData.setValue("resize", f"{width}x{height}")
        else:
            appData.setValue("resize", f"")
        width, height = self.CropWidthpSettings.value(), self.CropHeightSettings.value()
        appData.setValue("crop_width", width)
        appData.setValue("crop_height", height)
        if any((width, height)):
            appData.setValue("crop", f"{width}:{height}")
        else:
            appData.setValue("crop", f"")

        """Scene Detection"""
        appData.setValue("no_scdet", not self.ScedetChecker.isChecked())
        appData.setValue("scdet_mix", self.ScdetUseMix.isChecked())
        appData.setValue("use_fixed_scdet", self.UseFixedScdet.isChecked())
        appData.setValue("scdet_output", self.ScdetOutput.isChecked())
        appData.setValue("scdet_threshold", self.ScdetSelector.value())
        appData.setValue("fixed_max_scdet", self.ScdetMaxDiffSelector.value())
        appData.setValue("scdet_flow", self.ScdetFlowLen.currentIndex())
        appData.setValue("scdet_mode", self.ScdetMode.currentIndex())

        """Duplicate Frames Removal"""
        appData.setValue("remove_dup_mode", self.DupRmMode.currentIndex())
        appData.setValue("dup_threshold", self.DupFramesTSelector.value())

        """RAM Management"""
        appData.setValue("use_manual_buffer", self.MBufferChecker.isChecked())
        appData.setValue("manual_buffer_size", self.BufferSizeSelector.value())

        """Super Resolution Settings"""
        appData.setValue("use_sr", self.UseAiSR.isChecked())
        appData.setValue("use_sr_algo", self.AiSrSelector.currentText())
        appData.setValue("use_sr_model", self.AiSrModuleSelector.currentText())
        appData.setValue("use_sr_mode", self.AiSrMode.currentIndex())

        """RIFE Settings"""
        appData.setValue("ncnn", self.UseNCNNButton.isChecked())
        appData.setValue("ncnn_thread", self.ncnnInterpThreadCnt.value())
        appData.setValue("ncnn_gpu", self.ncnnSelectGPU.value())
        appData.setValue("tta_mode", self.TtaModeChecker.isChecked())
        appData.setValue("fp16", self.FP16Checker.isChecked())
        appData.setValue("scale", self.InterpScaleSelector.currentText())
        appData.setValue("reverse", self.ReverseChecker.isChecked())
        appData.setValue("selected_model", os.path.join(appData.value("model"), self.ModuleSelector.currentText()))
        appData.setValue("selected_model_name", self.ModuleSelector.currentText())
        appData.setValue("use_specific_gpu", self.DiscreteCardSelector.currentIndex())
        appData.setValue("auto_scale", self.AutoInterpScaleChecker.isChecked())
        appData.setValue("forward_ensemble", self.ForwardEnsembleChecker.isChecked())

        """Debug Mode"""
        appData.setValue("debug", self.DebugChecker.isChecked())

        """Preferences"""
        appData.setValue("multi_task_rest", self.multi_task_rest)
        appData.setValue("multi_task_rest_interval", self.multi_task_rest_interval)
        appData.setValue("after_mission", self.after_mission)
        appData.setValue("force_cpu", self.force_cpu)
        appData.setValue("expert_mode", self.expert_mode)

        """SVFI Main Page Position and Size"""
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))

        logger.info("[Main]: Download all settings")
        self.OptionCheck.isReadOnly = True
        appData.sync()
        pass

    def settings_check_args(self) -> bool:
        """
        Check are all args available
        :return:
        """
        videos = self.function_get_input_files()
        output_dir = self.OutputFolder.text()

        if not len(videos) or not len(output_dir):
            self.function_send_msg("Empty Input", "请输入要补帧的文件和输出文件夹")
            return False

        if len(videos) > 1:
            self.ProgressBarVisibleControl.setVisible(True)
        else:
            self.ProgressBarVisibleControl.setVisible(False)

        if not os.path.exists(output_dir):
            logger.info("Not Exists OutputFolder")
            self.function_send_msg("Output Folder Not Found", "输入文件或输出文件夹不存在！请确认输入")
            return False

        if os.path.isfile(output_dir):
            """Auto set Output Dir to correct form"""
            self.OutputFolder.setText(os.path.dirname(output_dir))

        for v in videos:
            if not os.path.exists(v):
                logger.info(f"Not Exists Input Source: {v}")
                self.function_send_msg("Input Source Not Found", f"输入文件:\n{v}\n不存在！请确认输入!")
                return False

        try:
            float(self.InputFPS.text())
            float(self.OutputFPS.text())
        except Exception:
            self.function_send_msg("Wrong Inputs", "请确认输入和输出帧率为有效数据")
            return False

        try:
            if self.slowmotion.isChecked():
                float(self.SlowmotionFPS.text())
        except Exception:
            self.function_send_msg("Wrong Inputs", "请确认慢动作输入帧率")
            return False

        return True

    def settings_set_start_info(self, start_frame, start_chunk, custom_prior=False):
        """
        设置启动帧数和区块信息
        :param custom_prior: Input is priority
        :param start_frame: StartFrame
        :param start_chunk: StartChunk
        :return: False: Custom Parameters Detected, no chunk removal is needed
        """
        if custom_prior:
            if self.StartFrame.value() != 0 and self.StartChunk != 1:
                return False
        self.StartFrame.setValue(start_frame)
        self.StartChunk.setValue(start_chunk)
        return True

    def settings_auto_set(self):
        """
        自动根据现有区块设置启动信息
        :return:
        """
        if not len(self.function_get_input_files()):
            return
        if self.InputFileName.currentItem() is None:
            self.InputFileName.setCurrentRow(0)
            # self.sendWarning("请选择", "请在左边输入栏选择要恢复进度的条目")
        project_dir = os.path.join(self.OutputFolder.text(),
                                   Utils.get_filename(self.InputFileName.currentItem().text()))
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
            self.settings_set_start_info(0, 1, True)
            return

        chunk_info_path = os.path.join(project_dir, "chunk.json")

        if not os.path.exists(chunk_info_path):
            logger.info("AutoSet find None to resume interpolation")
            self.settings_set_start_info(0, 1, True)
            return

        with open(chunk_info_path, "r", encoding="utf-8") as r:
            chunk_info = json.load(r)
        """
        key: project_dir, input filename, chunk cnt, chunk list, last frame
        """
        chunk_cnt = chunk_info["chunk_cnt"]
        last_frame = chunk_info["last_frame"]
        if chunk_cnt > 0:
            reply = self.function_send_msg(f"恢复进度？", f"检测到未完成的补帧任务，载入进度？", 3)
            if reply == QMessageBox.No:
                self.settings_set_start_info(0, 1, True)
                logger.info("User Abort Auto Set")
                return
        self.settings_set_start_info(last_frame + 1, chunk_cnt + 1, False)
        return

    def settings_update_gpu_info(self):
        infos = {}
        for i in range(torch.cuda.device_count()):
            card = torch.cuda.get_device_properties(i)
            info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
            infos[f"{i}"] = info
        logger.info(f"NVIDIA data: {infos}")

        if not len(infos):
            self.hasNVIDIA = False
            self.function_send_msg("No NVIDIA Card Found", "未找到N卡，将使用A卡或核显")
            appData.setValue("ncnn", True)
            self.UseNCNNButton.setChecked(True)
            self.UseNCNNButton.setEnabled(False)
            self.on_UseNCNNButton_clicked()
            return
        else:
            if self.UseNCNNButton.isChecked():
                appData.setValue("ncnn", True)
            else:
                appData.setValue("ncnn", False)

        self.DiscreteCardSelector.clear()
        for gpu in infos:
            self.DiscreteCardSelector.addItem(f"{gpu}: {infos[gpu]}")
        self.check_gpu = True
        return infos

    def settings_update_model_info(self):
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ncnn_dir = os.path.join(app_dir, "ncnn")
        rife_ncnn_dir = os.path.join(ncnn_dir, "rife")
        if self.UseNCNNButton.isChecked():
            model_dir = os.path.join(rife_ncnn_dir, "models")
        else:
            model_dir = os.path.join(app_dir, "train_log")
        appData.setValue("model", model_dir)

        if not os.path.exists(model_dir):
            logger.info(f"Not find Module dir at {model_dir}")
            self.function_send_msg("Model Dir Not Found", "未找到补帧模型路径，请检查！")
            return
        rife_model_list = list()
        for m in os.listdir(model_dir):
            if not os.path.isfile(os.path.join(model_dir, m)):
                rife_model_list.append(m)
        rife_model_list.reverse()
        self.ModuleSelector.clear()
        for mod in rife_model_list:
            self.ModuleSelector.addItem(f"{mod}")

    def settings_update_sr_algo(self):
        sr_ncnn_dir = self.function_get_SuperResolution_paths()

        if not os.path.exists(sr_ncnn_dir):
            logger.info(f"Not find SR Algorithm dir at {sr_ncnn_dir}")
            self.function_send_msg("Model Dir Not Found", "未找到补帧模型路径，请检查！")
            return

        algo_list = list()
        for m in os.listdir(sr_ncnn_dir):
            if not os.path.isfile(os.path.join(sr_ncnn_dir, m)):
                algo_list.append(m)

        self.AiSrSelector.clear()
        for algo in algo_list:
            self.AiSrSelector.addItem(f"{algo}")

    def settings_update_sr_model(self):
        """
        更新NCNN超分模型
        :return:
        """
        if not len(self.AiSrSelector.currentText()):
            return
        sr_algo_ncnn_dir = self.function_get_SuperResolution_paths(path_type=1,
                                                                   key_word=self.AiSrSelector.currentText())

        if not os.path.exists(sr_algo_ncnn_dir):
            logger.info(f"Not find SR Algorithm dir at {sr_algo_ncnn_dir}")
            self.function_send_msg("Model Dir Not Found", "未找到超分模型，请检查！")
            return

        model_list = list()
        for m in os.listdir(sr_algo_ncnn_dir):
            if not os.path.isfile(os.path.join(sr_algo_ncnn_dir, m)):
                model_list.append(m)

        self.AiSrModuleSelector.clear()
        for model in model_list:
            self.AiSrModuleSelector.addItem(f"{model}")

    def function_generate_log(self, mode=0):
        """
        生成日志并提示用户
        :param mode:0 Error Log 1 Settings Log
        :return:
        """
        status_check = "[导出设置预览]\n\n"
        for key in appData.allKeys():
            status_check += f"{key} => {appData.value(key)}\n"
        if mode == 0:
            status_check += "\n\n[设置信息]\n\n"
            status_check += self.OptionCheck.toPlainText()
            log_path = os.path.join(self.OutputFolder.text(), "log", f"{datetime.datetime.now().date()}.error.log")
        else:  # 1
            log_path = os.path.join(self.OutputFolder.text(), "log", f"{datetime.datetime.now().date()}.settings.log")

        log_path_dir = os.path.dirname(log_path)
        if not os.path.exists(log_path_dir):
            os.mkdir(log_path_dir)
        with open(log_path, "w", encoding="utf-8") as w:
            w.write(status_check)
        if not self.DebugChecker.isChecked():
            os.startfile(log_path_dir)

    def function_get_input_files(self):
        """
        获取输入文件路径列表
        :return:
        """
        widgetres = []
        count = self.InputFileName.count()
        for i in range(count):
            widgetres.append(self.InputFileName.item(i).text())
        return widgetres

    def function_send_msg(self, title, string, msg_type=1):
        """
        标准化输出界面提示信息
        TODO: Localization
        :param title:
        :param string:
        :param msg_type: 1 warning 2 info 3 question
        :return:
        """
        if self.silent:
            return
        QMessageBox.setWindowIcon(self, QIcon('svfi.png'))
        if msg_type == 1:
            reply = QMessageBox.warning(self,
                                        f"{title}",
                                        f"{string}",
                                        QMessageBox.Yes)
        elif msg_type == 2:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes)
        elif msg_type == 3:
            reply = QMessageBox.information(self,
                                            f"{title}",
                                            f"{string}",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            return
        return reply

    def function_select_file(self, filename, folder=False, _filter=None, multi=False):
        """
        用户选择文件
        :param filename:
        :param folder:
        :param _filter:
        :param multi:
        :return:
        """
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption="选取文件夹")
            return directory
        if multi:
            files = QFileDialog.getOpenFileNames(None, caption=f"选择{filename}", filter=_filter)
            return files[0]
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}", filter=_filter)
        return directory[0]

    def function_quick_concat(self):
        """
        快速合并
        :return:
        """
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.settings_load_current()
        if not input_v or not input_a or not output_v:
            self.function_send_msg("Parameters unfilled", "请填写输入或输出视频路径！")
            return

        ffmpeg_command = f"""
            {self.ffmpeg} -i {Utils.fillQuotation(input_a)} -i {Utils.fillQuotation(input_v)} 
            -map 1:v:0 -map 0:a:0 -c:v copy -c:a copy -shortest {Utils.fillQuotation(output_v)} -y
        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] concat {ffmpeg_command}")
        ps = subprocess.Popen(ffmpeg_command)
        ps.wait()
        self.function_send_msg("音视频合并操作完成！", f"请查收", msg_type=2)

    def function_update_gif_process(self, emit_json: str):
        """
        更新GIF进程信息
        :param emit_json:
        :return:
        """
        self.GifButton.setEnabled(True)
        emit_json = json.loads(emit_json)
        target_fps = emit_json.get("data", {"target_fps": appData.value("target_fps", 50)})["target_fps"]
        self.function_send_msg("GIF操作完成！", f'GIF帧率:{target_fps}', 2)

    def function_quick_gif(self):
        """
        快速生成GIF
        :return:
        """
        input_v = self.GifInput.text()
        output_v = self.GifOutput.text()
        self.settings_load_current()
        if not input_v or not output_v:
            self.function_send_msg("Parameters unfilled", "请填写输入或输出视频路径！")
            return
        if not appData.value("target_fps"):
            appData.setValue("target_fps", 50)
            logger.info("Not find output GIF fps, Auto set GIF output fps to 50 as it's smooth enough")
        target_fps = appData.value("target_fps", 50, type=float)
        # if target_fps > 50:
        #     target_fps = 50
        #     logger.info("Auto set GIF output fps to 50 as it's smooth enough")
        width = self.ResizeWidthSettings.value()
        height = self.ResizeHeightSettings.value()
        resize = f"scale={width}:{height},"
        if not all((width, height)):
            resize = ""
        ffmpeg_command = f"""{self.ffmpeg} -hide_banner -i {Utils.fillQuotation(input_v)} -r {target_fps}
                         -lavfi "{resize}split[s0][s1];
                         [s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=floyd_steinberg" 
                         {"-loop 0" if self.GifLoopChecker.isChecked() else ""}
                         {Utils.fillQuotation(output_v)} -y""".strip().strip("\n").replace("\n", "").replace("\\", "\\")

        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        self.GifButton.setEnabled(False)
        ps = subprocess.Popen(ffmpeg_command)
        ps.wait()
        self.GifButton.setEnabled(True)
        self.function_send_msg("GIF制作完成", f"GIF帧率：{target_fps}", 2)
        # GIF_Thread = SVFI_Run_Others(ffmpeg_command, 23333, data={"target_fps": target_fps})
        # GIF_Thread.run_signal.connect(self.update_gif_making)
        # GIF_Thread.start()

    def function_get_SuperResolution_paths(self, path_type=0, key_word=""):
        """
        获取超分路径
        :param key_word: should be module name
        :param path_type: 0: algo 1: module
        :return:
        """
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ncnn_dir = os.path.join(app_dir, "ncnn")
        sr_ncnn_dir = os.path.join(ncnn_dir, "sr")
        if path_type == 0:
            return sr_ncnn_dir
        if path_type == 1:
            return os.path.join(sr_ncnn_dir, key_word, "models")

    def process_update_rife(self, json_data):
        """
        Communicate with RIFE Thread
        :return:
        """

        def generate_error_log():
            self.function_generate_log(0)

        def remove_last_line():
            cursor = self.OptionCheck.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.OptionCheck.setTextCursor(cursor)

        def error_handle():
            now_text = self.OptionCheck.toPlainText()
            if self.current_failed:
                return
            if "Input File not valid" in now_text:
                self.function_send_msg("Inputs Failed", "你的输入文件有问题！请检查输入文件是否能播放，路径有无特殊字符", )
                self.current_failed = True
                return
            elif "JSON" in now_text:
                self.function_send_msg("Input File Failed", "文件信息读取失败，请确保软件和视频文件路径均为纯英文、无空格且无特殊字符", )
                self.current_failed = True
                return
            elif "ascii" in now_text:
                self.function_send_msg("Software Path Failure", "请把软件所在文件夹移到纯英文、无中文、无空格路径下", )
                self.current_failed = True
                return
            elif "CUDA out of memory" in now_text:
                self.function_send_msg("CUDA Failed", "你的显存不够啦！去清一下后台占用显存的程序，或者去'高级设置'降低视频分辨率/使用半精度模式/更换补帧模型~", )
                self.current_failed = True
                return
            elif "cudnn" in now_text.lower() and "fail" in now_text.lower():
                self.function_send_msg("CUDA Failed", "请前往官网更新驱动www.nvidia.cn/Download/index.aspx", )
                self.current_failed = True
                return
            elif "Concat Test Error" in now_text:
                self.function_send_msg("Concat Failed", "区块合并音轨测试失败，请检查输出文件格式是否支持源文件音频", )
                self.current_failed = True
                return
            elif "Broken Pipe" in now_text:
                self.function_send_msg("Render Failed", "请检查渲染设置，确保输出分辨率为偶数，尝试关闭硬件编码以解决问题", )
                self.current_failed = True
                return
            elif "error" in data.get("subprocess", "").lower():
                logger.error(f"[At the end of One Line Shot]: \n {data.get('subprocess')}")
                self.function_send_msg("Something Went Wrong", f"程序运行出现错误！\n{data.get('subprocess')}\n联系开发人员解决", )
                self.current_failed = True
                return

        data = json.loads(json_data)
        self.progressBar.setMaximum(int(data["cnt"]))
        self.progressBar.setValue(int(data["current"]))
        new_text = ""

        if len(data.get("notice", "")):
            new_text += data["notice"] + "\n"

        if len(data.get("subprocess", "")):
            dup_keys_list = ["Process at", "frame=", "matroska @"]
            if any([i in data["subprocess"] for i in dup_keys_list]):
                tmp = ""
                lines = data["subprocess"].splitlines()
                for line in lines:
                    if not any([i in line for i in dup_keys_list]):
                        tmp += line + "\n"
                if tmp.strip() == lines[-1].strip():
                    lines[-1] = ""
                data["subprocess"] = tmp + lines[-1]
                remove_last_line()
            new_text += data["subprocess"]

        for line in new_text.splitlines():
            line = html.escape(line)

            check_line = line.lower()
            if "process at" in check_line:
                add_line = f'<p><span style=" font-weight:600;">{line}</span></p>'
            elif "program finished" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#55aa00;">{line}</span></p>'
            elif "info" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#17C2FF;">{line}</span></p>'
            elif any([i in check_line for i in
                      ["error", "invalid", "incorrect", "critical", "fail", "can't", "can not"]]):
                if all([i not in check_line for i in
                        ["invalid dts", "incorrect timestamps"]]):
                    add_line = f'<p><span style=" font-weight:600; color:#ff0000;">{line}</span></p>'
                else:
                    add_line = f'<p><span>{line}</span></p>'
            elif "warn" in check_line:
                add_line = f'<p><span style=" font-weight:600; color:#ffaa00;">{line}</span></p>'
            # elif "duration" in line.lower():
            #     add_line = f'<p><span style=" font-weight:600; color:#550000;">{line}</span></p>'
            else:
                add_line = f'<p><span>{line}</span></p>'
            self.OptionCheck.append(add_line)

        if data["finished"]:
            """Error Handle"""
            returncode = data["returncode"]
            complete_msg = f"共 {data['cnt']} 个补帧任务\n"
            if returncode == 0 or "Program Finished" in self.OptionCheck.toPlainText() or (
                    returncode is not None and returncode > 3000):
                """What the fuck is SWIG?"""
                complete_msg += '成功！'
                os.startfile(self.OutputFolder.text())
            else:
                complete_msg += f'失败, 返回码：{returncode}\n请将弹出的文件夹内error.txt发送至交流群排疑，' \
                                f'并尝试前往高级设置恢复补帧进度'
                error_handle()
                generate_error_log()

            self.function_send_msg("任务完成", complete_msg, 2)
            self.ConcatAllButton.setEnabled(True)
            self.StartExtractButton.setEnabled(True)
            self.StartRenderButton.setEnabled(True)
            self.AllInOne.setEnabled(True)
            self.current_failed = False

        self.OptionCheck.moveCursor(QTextCursor.End)

    # @pyqtSlot(bool)
    def on_InputFileName_currentItemChanged(self):
        if self.InputFileName.currentItem() is None:
            if len(self.function_get_input_files()):
                self.InputFileName.setCurrentRow(-1)
            else:
                return
        if self.InputFileName.currentItem() is None:
            return
        text = self.InputFileName.currentItem().text().strip('"')
        """empty text"""
        if text == "" or not os.path.isfile(text):
            return
        input_fps = Utils.get_fps(text)
        self.InputFPS.setText(f"{input_fps:.5f}")
        if not len(self.OutputFPS.text()):
            try:
                exp = int(self.ExpSelecter.currentText()[1:])
                self.OutputFPS.setText(f"{input_fps * exp:.5f}")
            except Exception:
                pass
        self.maintain_sep_settings_buttons()
        return

    @pyqtSlot(bool)
    def on_InputButton_clicked(self):
        input_files = self.function_select_file('要补帧的视频', multi=True)
        if not len(input_files):
            return
        for f in input_files:
            self.InputFileName.addItem(f)
        if not len(self.OutputFolder.text()):
            self.OutputFolder.setText(os.path.dirname(input_files[0]))

    @pyqtSlot(bool)
    def on_InputDirButton_clicked(self):
        input_directory = self.function_select_file("要补帧的图片序列文件夹", folder=True)
        self.InputFileName.addItem(input_directory)
        return

    @pyqtSlot(bool)
    def on_OutputButton_clicked(self):
        folder = self.function_select_file('要输出项目的文件夹', folder=True)
        self.OutputFolder.setText(folder)

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        懒人式启动补帧按钮
        :return:
        """
        if not self.settings_check_args():
            return
        self.settings_auto_set()
        # self.on_EncoderSelector_currentTextChanged()  # update Encoders
        self.settings_load_current()  # update settings

        reply = self.function_send_msg("Confirm Start Info", f"补帧将会从区块[{self.StartChunk.text()}], "
                                                             f"起始帧[{self.StartFrame.text()}]启动。\n请确保上述两者皆不为空。"
                                                             f"是否执行补帧？", 3)
        if reply == QMessageBox.No:
            return
        self.AllInOne.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run()
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()

        self.thread = RIFE_thread
        update_text = f"""
                    [SVFI {self.version} 补帧操作启动]
                    显示“Program finished”则任务完成
                    如果遇到任何问题，请将命令行（黑色界面）、基础设置、高级设置和输出窗口截全图并联系开发人员解决，
                    群号在首页说明\n
                    第一个文件的输入帧率：{self.InputFPS.text()}， 输出帧率：{self.OutputFPS.text()}， 
                    启用慢动作：{self.slowmotion.isChecked()}， 慢动作帧率：{self.SlowmotionFPS.text()}

                    输出理想情况只有一行，如果遇到消息叠加，请将窗口拉长。

                    参数说明：
                    R: 当前渲染的帧数，C: 当前处理的帧数，S: 最近识别到的转场，SC：识别到的转场数量，
                    TAT：Task Acquire Time， 单帧任务获取时间，即任务阻塞时间，如果该值过大，请考虑增加虚拟内存
                    PT：Process Time，单帧任务处理时间，单帧补帧（+超分）花费时间
                    QL：Queue Length，任务队列长度

                    如果遇到卡顿，请直接右上角强制终止
                """
        if appData.value("ncnn", type=bool):
            update_text += "\n使用A卡或核显：True\n"

        self.OptionCheck.setText(update_text)
        self.current_failed = False

        self.tabWidget.setCurrentIndex(1)  # redirect to info page

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        """
        自动设置启动信息按钮（点我就完事了）
        :return:
        """
        if not len(self.function_get_input_files()) or not len(self.OutputFolder.text()):
            self.function_send_msg("Invalid Inputs", "请检查你的输入和输出文件夹")
            return
        if self.settings_check_args():
            self.settings_auto_set()

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        """
        快速合并音视频按钮
        :return:
        """
        if not self.ConcatInputV.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file('请输入要进行音视频合并的视频文件')
            self.ConcatInputV.setText(input_filename)
            self.ConcatInputA.setText(input_filename)
            self.OutputConcat.setText(
                os.path.join(os.path.dirname(input_filename), f"{Utils.get_filename(input_filename)}_concat.mp4"))
            return
        self.function_quick_concat()
        pass

    @pyqtSlot(bool)
    def on_GifButton_clicked(self):
        """
        快速制作GIF按钮
        :return:
        """
        if not self.GifInput.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file('请输入要制作成gif的视频文件')
            self.GifInput.setText(input_filename)
            self.GifOutput.setText(
                os.path.join(os.path.dirname(input_filename), f"{Utils.get_filename(input_filename)}.gif"))
            return
        self.function_quick_gif()
        pass

    @pyqtSlot(bool)
    def on_MBufferChecker_clicked(self):
        """
        使用自定义内存限制
        :return:
        """
        logger.info("Switch To Manual Assign Buffer Size Mode: %s" % self.MBufferChecker.isChecked())
        self.BufferSizeSelector.setEnabled(self.MBufferChecker.isChecked())

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self):
        input_files = self.function_get_input_files()
        if not len(input_files):
            return
        input_filename = input_files[0]
        input_fps = Utils.get_fps(input_filename)
        if input_fps:
            try:
                self.OutputFPS.setText(f"{input_fps * int(self.ExpSelecter.currentText()[1:]):.5f}")
            except Exception:
                self.function_send_msg("帧率输入有误", "请确认输入输出帧率为有效数据")

    @pyqtSlot(bool)
    def on_UseNCNNButton_clicked(self):
        if self.hasNVIDIA and self.UseNCNNButton.isChecked():
            reply = self.function_send_msg(f"确定使用NCNN？", f"你有N卡，确定使用A卡/核显？", 3)
            if reply == QMessageBox.Yes:
                logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
            else:
                self.UseNCNNButton.setChecked(False)
        else:
            logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
        bool_result = not self.UseNCNNButton.isChecked()
        self.FP16Checker.setEnabled(bool_result)
        self.DiscreteCardSelector.setEnabled(bool_result)
        self.ncnnInterpThreadCnt.setEnabled(not bool_result)
        self.ncnnSelectGPU.setEnabled(not bool_result)
        self.ForwardEnsembleChecker.setEnabled(bool_result)
        self.settings_update_model_info()
        # self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(bool)
    def on_UseAiSR_clicked(self):
        use_ai_sr = self.UseAiSR.isChecked()
        self.SrField.setVisible(use_ai_sr)
        if use_ai_sr:
            self.settings_update_sr_algo()
            self.settings_update_sr_model()

    @pyqtSlot(str)
    def on_AiSrSelector_currentTextChanged(self):
        self.settings_update_sr_model()

    @pyqtSlot(str)
    def on_ResizeTemplate_currentTextChanged(self):
        """
        自定义输出分辨率
        :return:
        """
        current_template = self.ResizeTemplate.currentText()

        input_files = self.function_get_input_files()
        sample_file = input_files[0]
        if not os.path.isfile(sample_file):
            self.function_send_msg("Input File not Video", "输入文件非视频，请手动输入需要的分辨率")
            return

        try:
            input_stream = cv2.VideoCapture(sample_file)
            width = input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        except Exception:
            logger.error(traceback.format_exc())
            height, width = 0, 0

        if "SD" in current_template:
            width, height = 480, 270
        elif "1080p" in current_template:
            width, height = 1920, 1080
        elif "4K" in current_template:
            width, height = 3840, 2160
        elif "8K" in current_template:
            width, height = 7680, 4320
        elif "%" in current_template:
            ratio = int(current_template[:-1]) / 100
            width, height = width * ratio, height * ratio
        self.ResizeWidthSettings.setValue(width)
        self.ResizeHeightSettings.setValue(height)

    @pyqtSlot(bool)
    def on_AutoInterpScaleChecker_clicked(self):
        """使用动态光流"""
        logger.info("Switch To Auto Scale Mode: %s" % self.AutoInterpScaleChecker.isChecked())
        bool_result = not self.AutoInterpScaleChecker.isChecked()
        self.InterpScaleSelector.setEnabled(bool_result)

    @pyqtSlot(bool)
    def on_slowmotion_clicked(self):
        self.SlowmotionFPS.setEnabled(self.slowmotion.isChecked())

    @pyqtSlot(bool)
    def on_UseFixedScdet_clicked(self):
        logger.info("Switch To FixedScdetThreshold Mode: %s" % self.UseFixedScdet.isChecked())

    @pyqtSlot(bool)
    def on_ScedetChecker_clicked(self):
        bool_result = self.ScedetChecker.isChecked()
        self.ScdetSelector.setVisible(bool_result)
        self.UseFixedScdet.setVisible(bool_result)
        self.ScdetMaxDiffSelector.setVisible(bool_result)
        self.ScdetUseMix.setVisible(bool_result)
        self.ScdetOutput.setVisible(bool_result)

    @pyqtSlot(str)
    def on_DupRmMode_currentTextChanged(self):
        self.DupFramesTSelector.setVisible(
            self.DupRmMode.currentIndex() == 1)  # Single Threshold Duplicated Frames Removal

    @pyqtSlot(bool)
    def on_UseEncodeThread_clicked(self):
        self.EncodeThreadSelector.setVisible(self.UseEncodeThread.isChecked())

    @pyqtSlot(str)
    def on_HwaccelSelector_currentTextChanged(self):
        logger.info("Switch To HWACCEL Mode: %s" % self.HwaccelSelector.currentText())
        check = self.HwaccelSelector.currentText() == "NVENC"
        self.HwaccelPresetLabel.setVisible(check)
        self.HwaccelPresetSelector.setVisible(check)
        encoders = EncodePresetAssemply.encoder[self.HwaccelSelector.currentText()]
        try:
            self.EncoderSelector.disconnect()
        except Exception:
            pass
        self.EncoderSelector.clear()
        self.EncoderSelector.currentTextChanged.connect(self.on_EncoderSelector_currentTextChanged)
        for e in encoders:
            self.EncoderSelector.addItem(e)
        self.EncoderSelector.setCurrentIndex(0)
        self.on_EncoderSelector_currentTextChanged()

    @pyqtSlot(str)
    def on_EncoderSelector_currentTextChanged(self):
        self.PresetSelector.clear()
        currentHwaccel = self.HwaccelSelector.currentText()
        currentEncoder = self.EncoderSelector.currentText()
        presets = EncodePresetAssemply.encoder[currentHwaccel][currentEncoder]
        for preset in presets:
            self.PresetSelector.addItem(preset)
        self.HwaccelPresetLabel.setVisible("NVENC" in currentHwaccel)
        self.HwaccelPresetSelector.setVisible("NVENC" in currentHwaccel)

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex in [2, 3]:
            """Step 3"""
            if tabIndex == 1:
                self.progressBar.setValue(0)
            logger.info("[Main]: Start Loading Settings")
            self.settings_load_current()

    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """

        :return:
        """
        self.settings_load_current()  # update settings
        self.ConcatAllButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(concat_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText(f"""
                    [SVFI {self.version} 仅合并操作启动，请移步命令行查看进度详情]
                    显示“Program finished”则任务完成
                    如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                    \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_StartExtractButton_clicked(self):
        """

        :return:
        """
        self.settings_load_current()
        self.StartExtractButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(extract_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText(f"""
                            [SVFI {self.version} 仅拆帧操作启动，请移步命令行查看进度详情]
                            显示“Program finished”则任务完成
                            如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                            \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_StartRenderButton_clicked(self):
        """

        :return:
        """
        self.settings_load_current()
        self.StartRenderButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(render_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.thread = RIFE_thread
        self.OptionCheck.setText(f"""
                            [SVFI {self.version} 仅渲染操作启动，请移步命令行查看进度详情]
                            显示“Program finished”则任务完成
                            如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                            \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_KillProcButton_clicked(self):
        """
        :return:
        """
        if self.thread is not None:
            self.thread.kill_proc_exec()

    @pyqtSlot(bool)
    def on_PauseProcess_clicked(self):
        """
        :return:
        """
        if self.thread is not None:
            self.thread.pause_proc_exec()
            if not self.pause:
                self.pause = True
                self.PauseProcess.setText("继续补帧！")
            else:
                self.pause = False
                self.PauseProcess.setText("暂停补帧！")

    def maintain_sep_settings_buttons(self):
        multi_input = len(self.function_get_input_files()) > 1
        self.SaveCurrentSettings.setVisible(multi_input)
        self.LoadCurrentSettings.setVisible(multi_input)

    @pyqtSlot(bool)
    def on_ShowAdvance_clicked(self):
        bool_result = self.AdvanceSettingsArea.isVisible()

        self.maintain_sep_settings_buttons()

        self.AdvanceSettingsArea.setVisible(not bool_result)
        if not bool_result:
            self.ShowAdvance.setText("隐藏高级设置")
        else:
            self.ShowAdvance.setText("显示高级设置")
        self.splitter.moveSplitter(10000000, 1)

    def SaveInputSettingsProcess(self, current_filename):
        config_maintainer = SVFI_Config_Manager()
        config_maintainer.DuplicateConfig(current_filename)

    @pyqtSlot(bool)
    def on_SaveCurrentSettings_clicked(self):
        self.settings_load_current()
        try:
            self.SaveInputSettingsProcess(self.InputFileName.currentItem().text())
            self.function_send_msg("Success", "当前输入设置保存成功，可直接补帧", 3)
        except Exception:
            self.function_send_msg("Save Failed", "请在输入列表选中要保存设置的条目")
        pass

    @pyqtSlot(bool)
    def on_LoadCurrentSettings_clicked(self):
        global appData
        config_maintainer = SVFI_Config_Manager()
        try:
            current_filename = self.InputFileName.currentItem().text()
            if not config_maintainer.MaintainConfig(current_filename):
                self.function_send_msg("Not Found Config", "未找到与当前输入匹配的设置文件")
                return
        except Exception:
            self.function_send_msg("Load Failed", "请在输入列表选中要加载设置的条目")
        appData = QSettings(appDataPath, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")
        self.settings_initiation()
        self.function_send_msg("Success", "已载入与当前输入匹配的设置", 3)
        pass

    @pyqtSlot(bool)
    def on_ClearInputButton_clicked(self):
        self.on_actionClearAllVideos_triggered()

    @pyqtSlot(bool)
    def on_OutputSettingsButton_clicked(self):
        self.function_generate_log(1)
        self.function_send_msg("Generate Success", "设置导出成功！settings.log即为设置快照", 3)
        pass

    @pyqtSlot(bool)
    def on_RefreshStartInfo_clicked(self):
        self.settings_set_start_info(0, 1)
        pass

    @pyqtSlot(bool)
    def on_actionManualGuide_triggered(self):
        SVFI_help_form = SVFI_Help_Dialog(self)
        SVFI_help_form.setWindowTitle("SVFI Quick Guide")
        SVFI_help_form.show()

    @pyqtSlot(bool)
    def on_actionAbout_triggered(self):
        SVFI_about_form = SVFI_About_Dialog(self)
        SVFI_about_form.setWindowTitle("About")
        SVFI_about_form.show()

    @pyqtSlot(bool)
    def on_actionPreferences_triggered(self):
        def generate_preference_dict():
            preference_dict = dict()
            preference_dict["multi_task_rest"] = self.multi_task_rest
            preference_dict["multi_task_rest_interval"] = self.multi_task_rest_interval
            preference_dict["after_mission"] = self.after_mission
            preference_dict["expert"] = self.expert_mode
            preference_dict["force_cpu"] = self.force_cpu
            return preference_dict

        self.SVFI_Preference_form = SVFI_Preference_Dialog(preference_dict=generate_preference_dict())
        self.SVFI_Preference_form.setWindowTitle("Preference")
        self.SVFI_Preference_form.preference_signal.connect(self.on_Preference_changed)
        self.SVFI_Preference_form.show()

    def on_Preference_changed(self, preference_dict: dict):

        self.multi_task_rest = preference_dict["multi_task_rest"]
        self.multi_task_rest_interval = preference_dict["multi_task_rest_interval"]
        self.after_mission = preference_dict["after_mission"]
        self.expert_mode = preference_dict["expert"]
        self.force_cpu = preference_dict["force_cpu"]
        self.on_ExpertMode_changed()

    def on_ExpertMode_changed(self):
        self.UseFixedScdet.setVisible(self.expert_mode)
        self.ScdetMaxDiffSelector.setVisible(self.expert_mode)
        self.HwaccelPresetSelector.setVisible(self.expert_mode)
        self.HwaccelPresetLabel.setVisible(self.expert_mode)
        self.QuickExtractChecker.setVisible(self.expert_mode)
        self.StrictModeChecker.setVisible(self.expert_mode)
        self.RenderSettingsLabel.setVisible(self.expert_mode)
        self.RenderSettingsGroup.setVisible(self.expert_mode)
        self.ReverseChecker.setVisible(self.expert_mode)
        # self.TtaModeChecker.setVisible(self.expert_mode)
        self.KeepChunksChecker.setVisible(self.expert_mode)
        self.AutoInterpScaleChecker.setVisible(self.expert_mode)

        if self.free:
            self.settings_free_hide()
        self.settings_dilapidation_hide()

    @pyqtSlot(bool)
    def on_actionImportVideos_triggered(self):
        self.on_InputButton_clicked()

    @pyqtSlot(bool)
    def on_actionStartProcess_triggered(self):
        if not self.AllInOne.isEnabled():
            self.function_send_msg("Invalid Operation", "已有任务在执行")
            return
        self.on_AllInOne_clicked()

    @pyqtSlot(bool)
    def on_actionStopProcess_triggered(self):
        self.on_KillProcButton_clicked()

    @pyqtSlot(bool)
    def on_actionClearVideo_triggered(self):
        try:
            currentIndex = self.InputFileName.currentIndex().row()
            self.InputFileName.takeItem(currentIndex)
        except Exception:
            self.function_send_msg("Fail to Clear Video", "未选中输入项")

    @pyqtSlot(bool)
    def on_actionQuit_triggered(self):
        sys.exit(0)

    @pyqtSlot(bool)
    def on_actionClearAllVideos_triggered(self):
        self.InputFileName.clear()
        self.maintain_sep_settings_buttons()

    @pyqtSlot(bool)
    def on_actionSaveSettings_triggered(self):
        self.on_SaveCurrentSettings_clicked()

    @pyqtSlot(bool)
    def on_actionLoadDefaultSettings_triggered(self):
        appData.clear()
        self.settings_initiation()
        self.hasNVIDIA = True
        self.settings_update_gpu_info()
        self.settings_update_model_info()
        self.on_EncoderSelector_currentTextChanged()  # Flush Encoder Sets
        self.function_send_msg("Load Success", "已载入默认设置", 3)

    def closeEvent(self, event):
        reply = self.function_send_msg("Quit", "是否保存当前设置？", 3)
        if reply == QMessageBox.Yes:
            self.settings_load_current()
            event.accept()
        else:
            event.ignore()
        pass


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        form = RIFE_GUI_BACKEND()
        form.show()
        app.exec_()
        sys.exit()
    except Exception:
        logger.critical(traceback.format_exc())
        sys.exit()
