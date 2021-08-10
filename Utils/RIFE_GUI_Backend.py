import datetime
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess as sp
import sys
import time
import traceback

import cv2
import psutil
import torch
from PyQt5.QtCore import QSettings, pyqtSignal, pyqtSlot, QThread, QTime, QVariant, QPoint, QSize
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication, QMessageBox, QFileDialog
from Utils import SVFI_UI, SVFI_help, SVFI_about, SVFI_preference, SVFI_preview_args
from Utils.RIFE_GUI_Custom import SVFI_Config_Manager
from Utils.utils import Tools, EncodePresetAssemply, ImgSeqIO, SupportFormat, ArgumentManager, SteamValidation

MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False

Utils = Tools()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(dname)

appDataPath = os.path.join(dname, "SVFI.ini")
appData = QSettings(appDataPath, QSettings.IniFormat)
appData.setIniCodec("UTF-8")

logger = Utils.get_logger("GUI", dname)
ols_potential = os.path.join(dname, "one_line_shot_args.exe")
ico_path = os.path.join(dname, "svfi.ico")


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


class SVFI_Preview_Args_Dialog(QDialog, SVFI_preview_args.Ui_Dialog):
    def __init__(self, parent=None):
        super(SVFI_Preview_Args_Dialog, self).__init__(parent)
        self.setWindowIcon(QIcon(ico_path))
        self.setupUi(self)
        self.default_args = self.ArgsLabel.text()
        self.args_list = re.findall("\{(.*?)\}", self.default_args)
        self.ArgumentsPreview()

    def ArgumentsPreview(self):
        """
        Generate String for Label of Arguments' Preview
        :return:
        """
        args_string = str(self.default_args)
        for arg_key in self.args_list:
            arg_data = appData.value(arg_key, "")
            _arg = ""
            if isinstance(arg_data, list):
                _arg = "\n".join(arg_data)
            else:
                _arg = str(arg_data)
            args_string = args_string.replace(f"{{{arg_key}}}", html.escape(_arg))

        logger.info(f"Check Arguments Preview: \n{args_string}")
        self.ArgsLabel.setText(args_string)


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
        self.ForceCpuChecker.setChecked(self.preference_dict["rife_use_cpu"])
        self.ExpertModeChecker.setChecked(self.preference_dict["expert"])
        self.PreviewArgsModeChecker.setChecked(self.preference_dict["is_preview_args"])
        self.QuietModeChecker.setChecked(self.preference_dict["is_gui_quiet"])
        self.OneWayModeChecker.setChecked(self.preference_dict["use_clear_inputs"])
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
        preference_dict["rife_use_cpu"] = self.ForceCpuChecker.isChecked()
        preference_dict["expert"] = self.ExpertModeChecker.isChecked()
        preference_dict["is_preview_args"] = self.PreviewArgsModeChecker.isChecked()
        preference_dict["is_gui_quiet"] = self.QuietModeChecker.isChecked()
        preference_dict["use_clear_inputs"] = self.OneWayModeChecker.isChecked()
        self.preference_signal.emit(preference_dict)


class SVFI_Run_Others(QThread):
    run_signal = pyqtSignal(dict)

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
        self.run_signal.emit(emit_json)

    def run(self):
        logger.info(f"[CMD Thread]: Start execute {self.command}")
        ps = Tools.popen(self.command)
        ps.wait()
        self.fire_finish_signal()
        pass

    pass


class SVFI_Run(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, parent=None, concat_only=False, extract_only=False, render_only=False):
        """
        Launch Task Thread
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
        self.current_filename = ""
        self.current_step = 0

    def build_command(self, item_data: dict) -> (str, str):
        global appData
        config_manager = SVFI_Config_Manager(item_data, dname)
        config_path = config_manager.FetchConfig()
        if config_path is None:
            logger.error(f"Invalid Task: {item_data}")
            return None, ""

        appData = QSettings(config_path, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")

        if os.path.splitext(appData.value("ols_path"))[-1] == ".exe":
            self.command = appData.value("ols_path") + " "
        else:
            self.command = f'python "{appData.value("ols_path")}" '

        input_path = item_data['input_path']
        task_id = item_data['task_id']

        """Some Additional Final Check"""
        if not len(input_path) or not os.path.exists(input_path):
            self.command = ""
            logger.error(f"Invalid Input: {item_data}")
            return None, ""

        if float(appData.value("input_fps", -1.0, type=float)) <= 0 or float(
                appData.value("target_fps", -1.0, type=float)) <= 0:
            logger.error(f"Invalid FPS/Target_FPS: {item_data}")
            return None, ""

        self.command += f'--input {Tools.fillQuotation(input_path)} --task-id {task_id} '

        output_path = appData.value("output_dir")
        if os.path.isfile(output_path):
            logger.info("OutputPath with FileName detected")
            appData.setValue("output_dir", os.path.dirname(output_path))

        self.command += f'--output {Tools.fillQuotation(output_path)} '
        self.command += f'--config {Tools.fillQuotation(config_path)} '

        """Alternative Mission Settings"""
        if self.concat_only:
            self.command += f"--concat-only "
        if self.extract_only:
            self.command += f"--extract-only "
        if self.render_only:
            self.command += f"--render-only "

        self.command = self.command.replace("\\", "/")
        return input_path, self.command

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
        appData.setValue("output_chunk_cnt", 1)
        appData.setValue("interp_start", 0)
        appData.setValue("input_start_point", "00:00:00")
        appData.setValue("input_end_point", "00:00:00")

    def run(self):
        try:
            logger.info("SVFI Task Run")
            try:
                input_list_data = json.loads(appData.value("gui_inputs", "{}"))
            except json.decoder.JSONDecodeError:
                logger.info("Failed to execute RIFE Tasks as there are no valid gui_inputs, check appData")
                self.update_status(True, returncode=502)
                return

            command_list = list()
            for item_data in input_list_data['inputs']:
                input_path, command = self.build_command(item_data)
                if not len(command):
                    continue
                command_list.append((input_path, command))

            self.current_step = 0
            self.task_cnt = len(command_list)

            if self.task_cnt > 1:
                """MultiTask"""
                appData.setValue("batch", True)

            if not self.task_cnt:
                logger.info("Task List Empty, Please Check Your Settings! (input fps for example)")
                self.update_status(True, "\nTask List is Empty!\n请点击输入条目以更新设置并确认输入输出帧率不为0", returncode=404)
                return

            interval_time = time.time()
            try:
                for input_path, command in command_list:
                    logger.info(f"Designed Command:\n{command}")
                    proc_args = shlex.split(command)
                    self.current_proc = sp.Popen(args=proc_args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding='utf-8',
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
                            self.update_status(False, notice=f"\n\nWARNING, 补帧已被手动暂停", returncode=0)

                            while True:
                                if self.kill:
                                    break
                                elif not self.pause:
                                    pause.resume()
                                    self.update_status(False, notice=f"\n\nWARNING, 补帧已继续",
                                                       returncode=0)
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
                    self.update_status(False, f"\nINFO - {datetime.datetime.now()} {input_path} 完成\n\n")
                    self.maintain_multitask()
                    # self.config_maintainer.RemoveConfig()

            except Exception:
                logger.error(traceback.format_exc())

            self.update_status(True, returncode=self.current_proc.returncode)
            logger.info("Tasks Finished")
            if self.current_proc.returncode == 0:
                """Finish Normally"""
                if appData.value("after_mission", type=int) == 1:
                    logger.info("Task Finished Normally, User Request to Shutdown")
                    os.system("shutdown -s -t 120")
                elif appData.value("after_mission", type=int) == 2:
                    logger.info("Task Finished Normally, User Request to Hibernate")
                    os.system("shutdown -h")
        except Exception:
            logger.error("Task Badly Finished", traceback.format_exc())
            self.update_status(True, returncode=1)
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


class RIFE_GUI_BACKEND(QMainWindow, SVFI_UI.Ui_MainWindow):
    kill_proc = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        """
        SVFI 主界面类初始化方法

        传参变量命名手册
        ;字符串或数值：类_功能或属性
        ;属性布尔：is_类_功能
        ;使能布尔：use_类_功能
        ;特殊布尔（单一成类）：类
        添加功能三步走：
        ！！！初始化用户选项载入->将现有界面的选项缓存至配置文件中->特殊配置！！！

        添加新选项/变量 3/3 实现先于load_current_settings的特殊新配置
        :param parent:
        """
        super(RIFE_GUI_BACKEND, self).__init__()
        self.setupUi(self)
        self.rife_thread = None
        self.chores_thread = None
        self.version = ArgumentManager.version_tag
        self.is_free = ArgumentManager.is_free
        self.is_steam = ArgumentManager.is_steam

        appData.setValue("app_dir", ddname)

        if appData.value("ffmpeg", "") != "ffmpeg":
            self.ffmpeg = f'"{os.path.join(appData.value("ffmpeg", ""), "ffmpeg.exe")}"'
        else:
            self.ffmpeg = appData.value("ffmpeg", "")

        # debug
        # self.ffmpeg = '"D:/60-fps-Project/Projects/RIFE GUI/release/SVFI.Env/神威 SVFI 3.2/Package/ffmpeg.exe"'

        if os.path.exists(appDataPath):
            logger.info("Previous Settings Found")

        self.check_gpu = False  # 是否检查过gpu
        self.current_failed = False  # 当前任务失败flag
        self.pause = False  # 当前任务是否暂停
        self.last_item = None  # 上一次点击的条目

        """Preference Maintainer"""
        self.multi_task_rest = False
        self.multi_task_rest_interval = 0
        self.after_mission = 0
        self.force_cpu = False
        self.expert_mode = True
        self.preview_args = False
        self.is_gui_quiet = False
        self.use_clear_inputs = False
        self.rife_cuda_cnt = 0
        self.SVFI_Preference_form = None
        self.resize_exp = 0

        """Initiate and Check GPU"""
        self.hasNVIDIA = True
        self.settings_update_pack()

        """Initiate Beautiful Layout and Signals"""
        self.AdvanceSettingsArea.setVisible(False)
        self.ProgressBarVisibleControl.setVisible(False)

        """Link InputFileName Event"""
        # self.InputFileName.clicked.connect(self.on_InputFileName_currentItemChanged)
        # self.InputFileName.itemClicked.connect(self.on_InputFileName_currentItemChanged)
        self.InputFileName.currentItemChanged.connect(self.on_InputFileName_currentItemChanged)
        self.InputFileName.addSignal.connect(self.on_InputFileName_currentItemChanged)

        """Dilapidation and Free Version Maintainer"""
        if self.is_free:
            self.settings_free_hide()
        self.settings_dilapidation_hide()

        self.STEAM = SteamValidation(self.is_steam, logger=logger)
        if self.is_steam:
            if not self.STEAM.steam_valid:
                warning_title = "Steam认证出错！SVFI用不了啦！"
                error = self.STEAM.steam_error
                logger.error(f"Steam Validation failed\n{error}")
                self.function_send_msg(warning_title, error)
            else:
                valid_response = self.STEAM.CheckSteamAuth()
                # debug
                # valid_response = 1
                if valid_response != 0:
                    self.STEAM.steam_valid = False
                    warning_title = "Steam认证失败！SVFI用不了啦！"
                    warning_msg = f"错误代码：{valid_response}"
                    if valid_response == 1:
                        warning_msg = "Ticket is not valid.\n白嫖党爬呀！"
                    elif valid_response == 2:
                        warning_msg = "A ticket has already been submitted for this steamID"
                    elif valid_response == 3:
                        warning_msg = "Ticket is from an incompatible interface version"
                    elif valid_response == 4:
                        warning_msg = "Ticket is not for this game\n白嫖怪爬呀！"
                    elif valid_response == 5:
                        warning_msg = "Ticket has expired\n购买的凭证过期"
                    self.function_send_msg(warning_title, warning_msg)
                    return

                if not self.is_free:
                    valid_response = self.STEAM.CheckProDLC()
                    if not valid_response:
                        self.STEAM.steam_valid = False
                        warning_title = "未购买专业版！SVFI用不了啦！"
                        warning_msg = f"请确保专业版DLC已安装"
                        self.function_send_msg(warning_title, warning_msg)
                        return

        os.chdir(dname)

    def settings_dilapidation_hide(self):
        """Hide Dilapidated Options"""
        self.ScdetModeLabel.setVisible(False)
        self.ScdetMode.setVisible(False)
        self.ScdetFlowLen.setVisible(False)
        self.SaveCurrentSettings.setVisible(False)
        self.LoadCurrentSettings.setVisible(False)
        self.SettingsPresetGroup.setVisible(False)
        self.LockWHChecker.setVisible(False)

    def settings_free_hide(self):
        """
        ST only
        :return:
        """
        # self.DupRmChecker.setVisible(False)
        self.DupFramesTSelector.setVisible(False)
        self.DupFramesTSelector.setValue(0.2)
        self.DupRmMode.clear()
        ST_RmMode = ["不去除重复帧", "单一识别", "去除一拍二", "去除一拍三"]
        for m in ST_RmMode:
            self.DupRmMode.addItem(m)

        self.StartPoint.setVisible(False)
        self.EndPoint.setVisible(False)
        self.StartPointLabel.setVisible(False)
        self.EndPointLabel.setVisible(False)
        self.ScdetOutput.setVisible(False)
        self.ScdetUseMix.setVisible(False)
        self.UseAiSR.setChecked(False)
        self.UseAiSR.setVisible(False)
        self.SrField.setVisible(False)
        self.RenderSettingsLabel.setVisible(False)
        self.RenderSettingsGroup.setVisible(False)
        # self.RenderOnlyGroupbox.setVisible(False)
        self.UseMultiCardsChecker.setVisible(False)
        self.TtaModeChecker.setVisible(False)
        self.DeinterlaceChecker.setVisible(False)
        self.FastDenoiseChecker.setVisible(False)
        self.HwaccelDecode.setVisible(False)
        self.EncodeThreadField.setVisible(False)
        self.HwaccelEncodeBox.setEnabled(False)

    def settings_update_pack(self, item_update=False):
        self.settings_initiation(item_update=item_update)
        self.settings_update_gpu_info()  # Flush GPU Info, 1
        self.on_UseNCNNButton_clicked(silent=True)  # G2
        self.settings_update_rife_model_info()
        self.on_HwaccelSelector_currentTextChanged()  # Flush Encoder Sets, 1
        self.on_EncoderSelector_currentTextChanged()  # E2
        self.on_UseEncodeThread_clicked()  # E3
        self.on_slowmotion_clicked()
        self.on_MBufferChecker_clicked()
        self.on_DupRmMode_currentTextChanged()
        self.on_ScedetChecker_clicked()
        self.on_ImgOutputChecker_clicked()
        self.on_AutoInterpScaleChecker_clicked()
        self.on_UseMultiCardsChecker_clicked()
        self.on_InterpExpReminder_toggled()
        self.on_UseAiSR_clicked()
        self.on_ResizeTemplate_activated()
        self.on_ExpertMode_changed()
        self.settings_initiation(item_update=item_update)
        pass

    def settings_initiation(self, item_update=False):
        """
        初始化用户选项载入
        从配置文件中读取上一次设置并初始化页面
        添加新选项/变量 1/3 appData -> Options
        :item_update: if inputs' current item changed, activate this
        :return:
        """
        if not item_update:
            """New Initiation of GUI"""
            try:
                input_list_data = json.loads(appData.value("gui_inputs", ""))
                if not len(self.function_get_input_paths()):
                    for item_data in input_list_data['inputs']:
                        config_maintainer = SVFI_Config_Manager(item_data, dname)
                        input_path = config_maintainer.FetchConfig()
                        if input_path is not None and os.path.exists(input_path):
                            self.InputFileName.addFileItem(item_data['input_path'],
                                                           item_data['task_id'])  # resume previous tasks
            except json.decoder.JSONDecodeError:
                logger.error("Could Not Find Valid GUI Inputs from appData, leave blank")

            """Maintain SVFI Startup Resolution"""
            desktop = QApplication.desktop()
            pos = appData.value("pos", QVariant(QPoint(960, 540)))
            size = appData.value("size", QVariant(QSize(int(desktop.width() * 0.6), int(desktop.height() * 0.5))))
            self.resize(size)
            self.move(pos)

        appData.setValue("ols_path", ols_potential)
        appData.setValue("ffmpeg", dname)

        if not os.path.exists(ols_potential):
            appData.setValue("ols_path",
                             r"D:\60-fps-Project\Projects\RIFE GUI\one_line_shot_args.py")
            appData.setValue("ffmpeg", "ffmpeg")
            logger.info("Change to Debug Path")

        """Basic Configuration"""
        self.OutputFolder.setText(appData.value("output_dir"))
        self.InputFPS.setText(appData.value("input_fps", "0"))
        self.OutputFPS.setText(appData.value("target_fps", ""))
        self.InterpExpReminder.setChecked(appData.value("is_exp_prior", False, type=bool))
        self.OutputFPSReminder.setChecked(not appData.value("is_exp_prior", False, type=bool))
        self.ExpSelecter.setCurrentText("x" + str(2 ** int(appData.value("rife_exp", "1"))))
        self.ImgOutputChecker.setChecked(appData.value("is_img_output", False, type=bool))
        appData.setValue("is_img_input", appData.value("is_img_input", False))
        appData.setValue("batch", False)
        self.KeepChunksChecker.setChecked(not appData.value("is_output_only", True, type=bool))
        self.StartPoint.setTime(QTime.fromString(appData.value("input_start_point", "00:00:00"), "HH:mm:ss"))
        self.EndPoint.setTime(QTime.fromString(appData.value("input_end_point", "00:00:00"), "HH:mm:ss"))
        self.StartChunk.setValue(appData.value("output_chunk_cnt", -1, type=int))
        self.StartFrame.setValue(appData.value("interp_start", -1, type=int))
        self.DebugChecker.setChecked(appData.value("debug", False, type=bool))

        """Output Resize Configuration"""
        self.CropHeightSettings.setValue(appData.value("crop_height", 0, type=int))
        self.CropWidthpSettings.setValue(appData.value("crop_width", 0, type=int))
        self.ResizeHeightSettings.setValue(appData.value("resize_height", 0, type=int))
        self.ResizeWidthSettings.setValue(appData.value("resize_width", 0, type=int))
        self.ResizeTemplate.setCurrentIndex(appData.value("resize_settings_index", 0, type=int))

        """Render Configuration"""
        self.UseCRF.setChecked(appData.value("use_crf", True, type=bool))
        self.CRFSelector.setValue(appData.value("render_crf", 16, type=int))
        self.UseTargetBitrate.setChecked(appData.value("use_bitrate", False, type=bool))
        self.BitrateSelector.setValue(appData.value("render_bitrate", 90, type=float))
        # self.PresetSelector.setCurrentText(appData.value("render_encoder_preset", "slow"))
        self.HwaccelSelector.setCurrentText(appData.value("render_hwaccel_mode", "CPU", type=str))
        self.HwaccelPresetSelector.setCurrentText(appData.value("render_hwaccel_preset", "None"))
        self.HwaccelDecode.setChecked(appData.value("use_hwaccel_decode", True, type=bool))
        self.UseEncodeThread.setChecked(appData.value("use_manual_encode_thread", False, type=bool))
        self.EncodeThreadSelector.setValue(appData.value("render_encode_thread", 16, type=int))
        self.EncoderSelector.setCurrentText(appData.value("render_encoder", "H264, 8bit"))
        self.PresetSelector.setCurrentText(appData.value("render_encoder_preset", "slow"))
        self.FFmpegCustomer.setText(appData.value("render_ffmpeg_customized", ""))
        self.ExtSelector.setCurrentText(appData.value("output_ext", "mp4"))
        self.RenderGapSelector.setValue(appData.value("render_gap", 1000, type=int))
        self.SaveAudioChecker.setChecked(appData.value("is_save_audio", True, type=bool))
        self.FastDenoiseChecker.setChecked(appData.value("use_fast_denoise", False, type=bool))
        self.StrictModeChecker.setChecked(appData.value("is_hdr_strict_mode", False, type=bool))
        self.QuickExtractChecker.setChecked(appData.value("is_quick_extract", True, type=bool))
        self.DeinterlaceChecker.setChecked(appData.value("use_deinterlace", False, type=bool))

        """Slowmotion Configuration"""
        self.slowmotion.setChecked(appData.value("is_render_slow_motion", False, type=bool))
        self.SlowmotionFPS.setText(appData.value("render_slow_motion_fps", "", type=str))
        self.GifLoopChecker.setChecked(appData.value("gif_loop", True, type=bool))

        """Scdet and RD Configuration"""
        self.ScedetChecker.setChecked(not appData.value("is_no_scdet", False, type=bool))
        self.ScdetSelector.setValue(appData.value("scdet_threshold", 12, type=int))
        self.ScdetUseMix.setChecked(appData.value("is_scdet_mix", False, type=bool))
        self.ScdetOutput.setChecked(appData.value("is_scdet_output", False, type=bool))
        self.ScdetFlowLen.setCurrentIndex(appData.value("scdet_flow_cnt", 0, type=int))
        self.UseFixedScdet.setChecked(appData.value("use_scdet_fixed", False, type=bool))
        self.ScdetMaxDiffSelector.setValue(appData.value("scdet_fixed_max", 40, type=int))
        self.ScdetMode.setCurrentIndex(appData.value("scdet_mode", 0, type=int))
        # self.DupRmChecker.setChecked(appData.value("remove_dup", False, type=bool))
        self.DupRmMode.setCurrentIndex(appData.value("remove_dup_mode", 0, type=int))
        self.DupFramesTSelector.setValue(appData.value("remove_dup_threshold", 10.00, type=float))

        """AI Super Resolution Configuration"""
        self.UseAiSR.setChecked(appData.value("use_sr", False, type=bool))
        self.SrTileSizeSelector.setValue(appData.value("sr_tilesize", 100, type=int))
        self.AiSrSelector.setCurrentText(appData.value("use_sr_algo", "realESR"))
        last_sr_model = appData.value("use_sr_model", "")
        if len(last_sr_model):
            self.AiSrModuleSelector.setCurrentText(last_sr_model)
        else:
            self.on_UseAiSR_clicked()
        self.AiSrMode.setCurrentIndex(appData.value("use_sr_mode", 0, type=int))

        """RIFE Configuration"""
        self.FP16Checker.setChecked(appData.value("use_rife_fp16", False, type=bool))
        self.InterpScaleSelector.setCurrentText(appData.value("rife_scale", "1.00"))
        self.ReverseChecker.setChecked(appData.value("is_rife_reverse", False, type=bool))
        self.ForwardEnsembleChecker.setChecked(appData.value("use_rife_forward_ensemble", False, type=bool))
        self.AutoInterpScaleChecker.setChecked(appData.value("use_rife_auto_scale", False, type=bool))
        self.on_AutoInterpScaleChecker_clicked()
        self.UseNCNNButton.setChecked(appData.value("use_ncnn", False, type=bool))
        self.TtaModeChecker.setChecked(appData.value("use_rife_tta_mode", False, type=bool))
        self.ncnnInterpThreadCnt.setValue(appData.value("ncnn_thread", 4, type=int))
        self.ncnnSelectGPU.setValue(appData.value("ncnn_gpu", 0, type=int))
        self.UseMultiCardsChecker.setChecked(appData.value("use_rife_multi_cards", False, type=bool))
        # Update RIFE Model
        rife_model_list = []
        for item_data in range(self.ModuleSelector.count()):
            rife_model_list.append(self.ModuleSelector.itemText(item_data))
        if appData.value("rife_model_name", "") in rife_model_list:
            self.ModuleSelector.setCurrentText(appData.value("rife_model_name", ""))

        """Multi Task Configuration"""
        self.multi_task_rest = appData.value("multi_task_rest", False, type=bool)
        self.multi_task_rest_interval = appData.value("multi_task_rest_interval", False, type=bool)
        self.after_mission = appData.value("after_mission", 0, type=int)
        self.force_cpu = appData.value("rife_use_cpu", False, type=bool)
        self.expert_mode = appData.value("expert_mode", True, type=bool)
        self.preview_args = appData.value("is_preview_args", False, type=bool)
        self.is_gui_quiet = appData.value("is_gui_quiet", False, type=bool)
        self.use_clear_inputs = appData.value("use_clear_inputs", False, type=bool)

        """REM Management Configuration"""
        self.MBufferChecker.setChecked(appData.value("use_manual_buffer", False, type=bool))
        self.BufferSizeSelector.setValue(appData.value("manual_buffer_size", 1, type=int))

        # self.setAttribute(Qt.WA_TranslucentBackground)

    def settings_load_current(self):
        """
        将现有界面的选项缓存至配置文件中
        添加新选项/变量 2/3 Options -> appData
        :return:
        """
        # """Restore AppData to root config"""
        # global appData
        # appData = QSettings(appDataPath, QSettings.IniFormat)
        # appData.setIniCodec("UTF-8")

        """Input Basic Input Information"""
        appData.setValue("version", self.version)
        appData.setValue("gui_inputs", self.InputFileName.saveTasks())
        appData.setValue("output_dir", self.OutputFolder.text())
        appData.setValue("input_fps", self.InputFPS.text())
        appData.setValue("target_fps", self.OutputFPS.text())
        appData.setValue("is_exp_prior", self.InterpExpReminder.isChecked())
        appData.setValue("rife_exp", int(math.log(int(self.ExpSelecter.currentText()[1:]), 2)))
        appData.setValue("is_img_output", self.ImgOutputChecker.isChecked())
        appData.setValue("is_output_only", not self.KeepChunksChecker.isChecked())
        appData.setValue("is_save_audio", self.SaveAudioChecker.isChecked())
        appData.setValue("output_ext", self.ExtSelector.currentText())

        """Input Time Stamp"""
        appData.setValue("input_start_point", self.StartPoint.time().toString("HH:mm:ss"))
        appData.setValue("input_end_point", self.EndPoint.time().toString("HH:mm:ss"))
        appData.setValue("output_chunk_cnt", self.StartChunk.value())
        appData.setValue("interp_start", self.StartFrame.value())
        appData.setValue("render_gap", self.RenderGapSelector.value())

        """Render"""
        appData.setValue("use_crf", self.UseCRF.isChecked())
        appData.setValue("use_bitrate", self.UseTargetBitrate.isChecked())
        appData.setValue("render_crf", self.CRFSelector.value())
        appData.setValue("render_bitrate", self.BitrateSelector.value())
        appData.setValue("render_encoder_preset", self.PresetSelector.currentText())
        appData.setValue("render_encoder", self.EncoderSelector.currentText())
        appData.setValue("render_hwaccel_mode", self.HwaccelSelector.currentText())
        appData.setValue("render_hwaccel_preset", self.HwaccelPresetSelector.currentText())
        appData.setValue("use_hwaccel_decode", self.HwaccelDecode.isChecked())
        appData.setValue("use_manual_encode_thread", self.UseEncodeThread.isChecked())
        appData.setValue("render_encode_thread", self.EncodeThreadSelector.value())
        appData.setValue("is_quick_extract", self.QuickExtractChecker.isChecked())
        appData.setValue("is_hdr_strict_mode", self.StrictModeChecker.isChecked())
        appData.setValue("render_ffmpeg_customized", self.FFmpegCustomer.text())
        appData.setValue("no_concat", False)  # always concat
        appData.setValue("use_fast_denoise", self.FastDenoiseChecker.isChecked())

        """Special Render Effect"""
        appData.setValue("gif_loop", self.GifLoopChecker.isChecked())
        appData.setValue("is_render_slow_motion", self.slowmotion.isChecked())
        appData.setValue("render_slow_motion_fps", self.SlowmotionFPS.text())
        if appData.value("is_render_slow_motion", False, type=bool):
            appData.setValue("is_save_audio", False)
            self.SaveAudioChecker.setChecked(False)
        appData.setValue("use_deinterlace", self.DeinterlaceChecker.isChecked())

        appData.setValue("resize_settings_index", self.ResizeTemplate.currentIndex())
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
        appData.setValue("is_no_scdet", not self.ScedetChecker.isChecked())
        appData.setValue("is_scdet_mix", self.ScdetUseMix.isChecked())
        appData.setValue("use_scdet_fixed", self.UseFixedScdet.isChecked())
        appData.setValue("is_scdet_output", self.ScdetOutput.isChecked())
        appData.setValue("scdet_threshold", self.ScdetSelector.value())
        appData.setValue("scdet_fixed_max", self.ScdetMaxDiffSelector.value())
        appData.setValue("scdet_flow_cnt", self.ScdetFlowLen.currentIndex())
        appData.setValue("scdet_mode", self.ScdetMode.currentIndex())

        """Duplicate Frames Removal"""
        appData.setValue("remove_dup_mode", self.DupRmMode.currentIndex())
        appData.setValue("remove_dup_threshold", self.DupFramesTSelector.value())

        """RAM Management"""
        appData.setValue("use_manual_buffer", self.MBufferChecker.isChecked())
        appData.setValue("manual_buffer_size", self.BufferSizeSelector.value())

        """Super Resolution Settings"""
        appData.setValue("use_sr", self.UseAiSR.isChecked())
        appData.setValue("use_sr_algo", self.AiSrSelector.currentText())
        appData.setValue("use_sr_model", self.AiSrModuleSelector.currentText())
        appData.setValue("use_sr_mode", self.AiSrMode.currentIndex())
        appData.setValue("sr_tilesize", self.SrTileSizeSelector.value())
        appData.setValue("resize_exp", self.resize_exp)

        """RIFE Settings"""
        appData.setValue("use_ncnn", self.UseNCNNButton.isChecked())
        appData.setValue("ncnn_thread", self.ncnnInterpThreadCnt.value())
        appData.setValue("ncnn_gpu", self.ncnnSelectGPU.value())
        appData.setValue("use_rife_tta_mode", self.TtaModeChecker.isChecked())
        appData.setValue("use_rife_fp16", self.FP16Checker.isChecked())
        appData.setValue("rife_scale", self.InterpScaleSelector.currentText())
        appData.setValue("is_rife_reverse", self.ReverseChecker.isChecked())
        appData.setValue("rife_model",
                         os.path.join(appData.value("rife_model_dir", ""), self.ModuleSelector.currentText()))
        appData.setValue("rife_model_name", self.ModuleSelector.currentText())
        appData.setValue("rife_cuda_cnt", self.rife_cuda_cnt)
        appData.setValue("use_rife_multi_cards", self.UseMultiCardsChecker.isChecked())
        appData.setValue("use_specific_gpu", self.DiscreteCardSelector.currentIndex())
        appData.setValue("use_rife_auto_scale", self.AutoInterpScaleChecker.isChecked())
        appData.setValue("use_rife_forward_ensemble", self.ForwardEnsembleChecker.isChecked())

        """Debug Mode"""
        appData.setValue("debug", self.DebugChecker.isChecked())

        """Preferences"""
        appData.setValue("multi_task_rest", self.multi_task_rest)
        appData.setValue("multi_task_rest_interval", self.multi_task_rest_interval)
        appData.setValue("after_mission", self.after_mission)
        appData.setValue("rife_use_cpu", self.force_cpu)
        appData.setValue("expert_mode", self.expert_mode)
        appData.setValue("is_preview_args", self.preview_args)
        appData.setValue("is_gui_quiet", self.is_gui_quiet)
        appData.setValue("use_clear_inputs", self.use_clear_inputs)

        """SVFI Main Page Position and Size"""
        appData.setValue("pos", QVariant(self.pos()))
        appData.setValue("size", QVariant(self.size()))

        logger.info("[Main]: Download all settings")
        self.OptionCheck.isReadOnly = True
        # if not os.path.exists(appData.fileName()):
        appData.sync()
        try:
            if not os.path.samefile(appData.fileName(), appDataPath):
                shutil.copy(appData.fileName(), appDataPath)
        except FileNotFoundError:
            logger.info("Unable to save Configs, probably permanent loss")
        pass

    def settings_check_args(self) -> bool:
        """
        Check are all args available
        :return:
        """
        input_paths = self.function_get_input_paths()
        output_dir = self.OutputFolder.text()

        if not len(input_paths) or not len(output_dir):
            self.function_send_msg("Empty Input", "请输入要补帧的文件和输出文件夹")
            return False

        if len(input_paths) > 1:
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

        for path in input_paths:
            if not os.path.exists(path):
                logger.info(f"Not Exists Input Source: {path}")
                self.function_send_msg("Input Source Not Found", f"输入文件:\n{path}\n不存在！请确认输入!")
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
        if not self.settings_check_args():
            return
        if not len(self.function_get_input_paths()):
            return
        current_item = self.InputFileName.currentItem()
        if current_item is None:
            self.function_send_msg(f"恢复进度？", f"正在使用队列的第一个任务进行进度检测")
            self.InputFileName.setCurrentRow(0)
            current_item = self.InputFileName.currentItem()
        output_dir = self.OutputFolder.text()

        widget_data = self.InputFileName.getWidgetData(current_item)
        input_path = widget_data.get('input_path')
        task_id = widget_data.get('task_id')
        project_dir = os.path.join(output_dir,
                                   f"{Tools.get_filename(input_path)}_{task_id}")
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
            self.function_send_msg(f"恢复进度", f"未找到与第{widget_data['row']}个任务相关的进度信息", 3)
            self.settings_set_start_info(0, 1, True)  # start from zero
            return

        if self.ImgOutputChecker.isChecked():
            """Img Output"""
            img_io = ImgSeqIO(logger=logger, folder=output_dir, is_tool=True, output_ext=self.ExtSelector.currentText())
            last_img = img_io.get_start_frame()  # output_dir
            if last_img:
                reply = self.function_send_msg(f"恢复进度？", f"检测到未完成的图片序列补帧任务，载入进度？", 3)
                if reply == QMessageBox.No:
                    self.settings_set_start_info(0, 1, False)  # start from zero
                    logger.info("User Abort Auto Set")
                    return
            self.settings_set_start_info(last_img + 1, 1, False)
            return

        chunk_info_path = os.path.join(project_dir, "chunk.json")

        if not os.path.exists(chunk_info_path):
            self.function_send_msg(f"恢复进度", f"未找到与第{widget_data['row']}个任务相关的进度信息", 3)
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
                self.settings_set_start_info(0, 1, False)
                logger.info("User Abort Auto Set")
                return
        self.settings_set_start_info(last_frame + 1, chunk_cnt + 1, False)
        return

    def settings_update_gpu_info(self):
        cuda_infos = {}
        self.rife_cuda_cnt = torch.cuda.device_count()
        for i in range(self.rife_cuda_cnt):
            card = torch.cuda.get_device_properties(i)
            info = f"{card.name}, {card.total_memory / 1024 ** 3:.1f} GB"
            cuda_infos[f"{i}"] = info
        logger.info(f"NVIDIA data: {cuda_infos}")

        if not len(cuda_infos):
            self.hasNVIDIA = False
            self.function_send_msg("No NVIDIA Card Found", "未找到N卡，将使用A卡或核显")
            appData.setValue("use_ncnn", True)
            self.UseNCNNButton.setChecked(True)
            self.UseNCNNButton.setEnabled(False)
            self.on_UseNCNNButton_clicked()
            return
        else:
            if self.UseNCNNButton.isChecked():
                appData.setValue("use_ncnn", True)
            else:
                appData.setValue("use_ncnn", False)

        self.DiscreteCardSelector.clear()
        for gpu in cuda_infos:
            self.DiscreteCardSelector.addItem(f"{gpu}: {cuda_infos[gpu]}")
        self.check_gpu = True
        return cuda_infos

    def settings_update_rife_model_info(self):
        app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ncnn_dir = os.path.join(app_dir, "ncnn")
        rife_ncnn_dir = os.path.join(ncnn_dir, "rife")
        if self.UseNCNNButton.isChecked():
            model_dir = os.path.join(rife_ncnn_dir, "models")
        else:
            model_dir = os.path.join(app_dir, "train_log")
        appData.setValue("rife_model_dir", model_dir)

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
        current_sr_algo = self.AiSrSelector.currentText()
        if not len(current_sr_algo):
            return
        sr_algo_ncnn_dir = self.function_get_SuperResolution_paths(path_type=1,
                                                                   key_word=current_sr_algo)

        if not os.path.exists(sr_algo_ncnn_dir):
            logger.info(f"Not find SR Algorithm dir at {sr_algo_ncnn_dir}")
            self.function_send_msg("Model Dir Not Found", "未找到超分模型，请检查！")
            return

        model_list = list()
        for m in os.listdir(sr_algo_ncnn_dir):
            if not os.path.isfile(os.path.join(sr_algo_ncnn_dir, m)):
                model_list.append(m)
            if "realESR" in current_sr_algo:
                # pth model only
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
        preview_args = SVFI_Preview_Args_Dialog(self).ArgsLabel.text()
        preview_args = html.unescape("\n".join(re.findall('">(.*?)</span>', preview_args)))
        status_check = f"[导出设置预览]\n\n{preview_args}\n\n"
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

    def function_get_input_paths(self):
        """
        获取输入文件路径列表
        :return:
        """
        widgetres = []
        count = self.InputFileName.count()
        for i in range(count):
            try:
                widgetres.append(self.InputFileName.itemWidget(self.InputFileName.item(i)).input_path)
            except:
                pass
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
        if self.is_gui_quiet:
            return
        QMessageBox.setWindowIcon(self, QIcon(ico_path))
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
            {self.ffmpeg} -i {Tools.fillQuotation(input_a)} -i {Tools.fillQuotation(input_v)} 
            -map 1:v:0 -map 0:a:0 -c:v copy -c:a copy -shortest {Tools.fillQuotation(output_v)} -y
        """.strip().strip("\n").replace("\n", "").replace("\\", "/")
        logger.info(f"[GUI] concat {ffmpeg_command}")
        self.chores_thread = SVFI_Run_Others(ffmpeg_command, data={"type": "音视频合并"})
        self.chores_thread.run_signal.connect(self.function_update_chores_finish)
        self.chores_thread.start()
        self.ConcatButton.setEnabled(False)

    def function_update_chores_finish(self, data: dict):
        mission_type = data['data']['type']
        self.function_send_msg("Chores Mission", f"{mission_type}任务完成", msg_type=2)
        self.ConcatButton.setEnabled(True)
        self.GifButton.setEnabled(True)

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

        width = self.ResizeWidthSettings.value()
        height = self.ResizeHeightSettings.value()
        resize = f"scale={width}:{height},"
        if not all((width, height)):
            resize = ""
        ffmpeg_command = f"""{self.ffmpeg} -hide_banner -i {Tools.fillQuotation(input_v)} -r {target_fps}
                         -lavfi "{resize}split[s0][s1];
                         [s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=floyd_steinberg" 
                         {"-loop 0" if self.GifLoopChecker.isChecked() else ""}
                         {Tools.fillQuotation(output_v)} -y""".strip().strip("\n").replace("\n", "").replace("\\", "\\")

        logger.info(f"[GUI] create gif: {ffmpeg_command}")
        self.chores_thread = SVFI_Run_Others(ffmpeg_command, data={"type": "GIF制作"})
        self.chores_thread.run_signal.connect(self.function_update_chores_finish)
        self.chores_thread.start()
        self.GifButton.setEnabled(False)

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
            dup_keys_list = ["Process at", "frame=", "matroska @", "0%|", f"{ArgumentManager.app_id}", "Steam ID",
                             "AppID"]
            if any([i in data["subprocess"] for i in dup_keys_list]):
                tmp = ""
                lines = data["subprocess"].splitlines()
                for line in lines:
                    if not any([i in line for i in dup_keys_list]) and len(line.strip()):
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
            self.InputFileName.setEnabled(True)
            self.current_failed = False
            self.InputFileName.refreshTasks()
            self.on_InputFileName_currentItemChanged()

            if self.use_clear_inputs:
                self.InputFileName.clear()

        self.OptionCheck.moveCursor(QTextCursor.End)

    # @pyqtSlot(bool)
    def on_InputFileName_currentItemChanged(self):
        current_item = self.InputFileName.currentItem()
        if current_item is None:
            item_count = len(self.InputFileName.getItems())
            if item_count:
                # self.InputFileName.setCurrentRow(item_count - 1)
                # TODO check impact
                current_item = self.InputFileName.currentItem()
            else:
                return
        if current_item is None:
            return
        if self.InputFileName.itemWidget(current_item) is None:  # check if exists this item in InputFileName
            return
        widget_data = self.InputFileName.getWidgetData(current_item)
        # previous_config_manager = SVFI_Config_Manager(widget_data, dname, logger)
        # if self.last_item != widget_data or previous_config_manager.FetchConfig() is None:
        if widget_data is not None:
            self.settings_maintain_item_settings(widget_data)  # 保存当前设置，并准备跳转到新任务的历史设置（可能没有）
        input_path = widget_data.get('input_path')
        input_fps = Tools.get_fps(input_path)
        if not len(self.InputFPS.text()):
            self.InputFPS.setText("0")
        if os.path.isfile(input_path):
            self.InputFPS.setText(f"{input_fps:.5f}")
            if os.path.isfile(input_path):
                ext = os.path.splitext(input_path)[1]
                if ext in SupportFormat.vid_outputs:
                    # self.ExtSelector.setCurrentText(ext.strip("."))
                    # TODO Check lock here
                    pass
        if self.InterpExpReminder.isChecked():  # use exp to calculate outputfps
            try:
                exp = int(self.ExpSelecter.currentText()[1:])
                self.OutputFPS.setText(f"{input_fps * exp:.5f}")
            except Exception:
                pass
        return

    @pyqtSlot(bool)
    def on_InputButton_clicked(self):
        input_files = self.function_select_file('要补帧的视频', multi=True)
        if not len(input_files):
            return
        for f in input_files:
            self.InputFileName.addFileItem(f)
        if not len(self.OutputFolder.text()):
            self.OutputFolder.setText(os.path.dirname(input_files[0]))

    @pyqtSlot(bool)
    def on_InputDirButton_clicked(self):
        input_directory = self.function_select_file("要补帧的图片序列文件夹", folder=True)
        self.InputFileName.addFileItem(input_directory)
        if not len(self.OutputFolder.text()):
            self.OutputFolder.setText(os.path.dirname(input_directory))
        return

    @pyqtSlot(bool)
    def on_OutputButton_clicked(self):
        folder = self.function_select_file('要输出项目的文件夹', folder=True)
        self.OutputFolder.setText(folder)

    def function_select_first_input(self):
        for it in range(len(self.function_get_input_paths())):
            self.InputFileName.setCurrentRow(it)

    @pyqtSlot(bool)
    def on_AllInOne_clicked(self):
        """
        懒人式启动补帧按钮
        :return:
        """
        self.function_select_first_input()
        if not self.settings_check_args():
            return
        # self.settings_auto_set()
        self.settings_load_current()  # update settings

        if self.preview_args and not self.is_gui_quiet:
            SVFI_preview_args_form = SVFI_Preview_Args_Dialog(self)
            SVFI_preview_args_form.setWindowTitle("Preview SVFI Arguments")
            # SVFI_preview_args_form.setWindowModality(Qt.ApplicationModal)
            SVFI_preview_args_form.exec_()
        reply = self.function_send_msg("Confirm Start Info", f"补帧将会从区块[{self.StartChunk.text()}], "
                                                             f"起始帧[{self.StartFrame.text()}]启动。\n请确保上述两者皆不为空。"
                                                             f"是否执行补帧？", 3)
        if reply == QMessageBox.No:
            return
        self.AllInOne.setEnabled(False)
        self.InputFileName.setEnabled(False)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run()
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()

        self.rife_thread = RIFE_thread
        update_text = f"""
                    [SVFI {self.version} 补帧操作启动]
                    显示“Program finished”则任务完成
                    如果遇到任何问题，请将基础设置、输出窗口截全图，不要录屏，并导出当前设置为settings.log文件并联系开发人员解决，
                    群号在首页说明\n
                    
                    参数说明：
                    R: 当前渲染的帧数，C: 当前处理的帧数，S: 最近识别到的转场，SC：识别到的转场数量，
                    TAT：Task Acquire Time， 单帧任务获取时间，即任务阻塞时间，如果该值过大，请考虑增加虚拟内存
                    PT：Process Time，单帧任务处理时间，单帧补帧（+超分）花费时间
                    QL：Queue Length，任务队列长度

                    如果遇到卡顿或软件卡死，请直接右上角强制终止
                """
        self.OptionCheck.setText(update_text)
        self.current_failed = False
        self.tabWidget.setCurrentIndex(1)  # redirect to info page

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        """
        自动设置启动信息按钮（点我就完事了）
        :return:
        """
        self.function_select_first_input()
        if self.InputFileName.currentItem() is None or not len(self.OutputFolder.text()):
            self.function_send_msg("Invalid Inputs", "请检查你的输入和输出文件夹")
            return
        self.settings_auto_set()

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
        if not self.InterpExpReminder.isChecked():
            return
        input_fps = self.InputFPS.text()
        if len(input_fps):
            try:
                self.OutputFPS.setText(f"{float(input_fps) * int(self.ExpSelecter.currentText()[1:]):.5f}")
            except Exception:
                self.function_send_msg("帧率输入有误", "请确认输入输出帧率为有效数据")

    @pyqtSlot(bool)
    def on_InterpExpReminder_toggled(self):
        bool_result = not self.InterpExpReminder.isChecked()
        self.OutputFPS.setEnabled(bool_result)
        if not bool_result:
            self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(bool)
    def on_UseNCNNButton_clicked(self, clicked=True, silent=False):
        if self.hasNVIDIA and self.UseNCNNButton.isChecked() and not silent:
            reply = self.function_send_msg(f"确定使用NCNN？", f"你有N卡，确定使用A卡/核显？", 3)
            if reply == QMessageBox.Yes:
                logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
            else:
                self.UseNCNNButton.setChecked(False)
        else:
            logger.info("Switch To NCNN Mode: %s" % self.UseNCNNButton.isChecked())
        bool_result = not self.UseNCNNButton.isChecked()
        self.AutoInterpScaleChecker.setEnabled(bool_result)
        self.on_AutoInterpScaleChecker_clicked()
        self.FP16Checker.setEnabled(bool_result)
        self.DiscreteCardSelector.setEnabled(bool_result)
        self.ncnnInterpThreadCnt.setEnabled(not bool_result)
        self.ncnnSelectGPU.setEnabled(not bool_result)
        self.ForwardEnsembleChecker.setEnabled(bool_result)
        self.settings_update_rife_model_info()
        # self.on_ExpSelecter_currentTextChanged()

    @pyqtSlot(bool)
    def on_UseMultiCardsChecker_clicked(self):
        bool_result = self.UseMultiCardsChecker.isChecked()
        self.DiscreteCardSelector.setEnabled(not bool_result)
        self.SelectedGpuLabel.setText("选择的GPU" if not bool_result else "（使用A卡或核显）拥有的GPU个数")

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

    @pyqtSlot(int)
    def on_ResizeTemplate_activated(self):
        """
        自定义输出分辨率
        :return:
        """
        current_template = self.ResizeTemplate.currentText()

        if not len(self.InputFileName.getItems()):
            return
        current_item = self.InputFileName.currentItem()
        if current_item is None:
            self.InputFileName.setCurrentRow(0)
            current_item = self.InputFileName.currentItem()
        row = self.InputFileName.getWidgetData(current_item)['row']
        input_files = self.function_get_input_paths()
        sample_file = input_files[row]
        # if not os.path.isfile(sample_file):
        #     self.function_send_msg("Input File not Video", "输入文件非视频，请手动输入需要的分辨率")
        #     return

        try:
            if not os.path.isfile(sample_file):
                height, width = 0, 0
            else:
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
            self.resize_exp = ratio
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
        self.on_ExpertMode_changed()
        if self.is_free:
            self.settings_free_hide()

    @pyqtSlot(str)
    def on_DupRmMode_currentTextChanged(self):
        self.DupFramesTSelector.setVisible(
            self.DupRmMode.currentIndex() == 1)  # Single Threshold Duplicated Frames Removal

    @pyqtSlot(bool)
    def on_ImgOutputChecker_clicked(self):
        """
        Support PNG or TIFF
        :return:
        """
        self.ExtSelector.clear()
        if self.ImgOutputChecker.isChecked():
            self.SaveAudioChecker.setChecked(False)
            self.SaveAudioChecker.setEnabled(False)
            for ext in SupportFormat.img_outputs:
                self.ExtSelector.addItem(ext.strip('.'))
        else:
            self.SaveAudioChecker.setEnabled(True)
            for ext in SupportFormat.vid_outputs:
                self.ExtSelector.addItem(ext.strip('.'))

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
    def on_GifButton_clicked(self):
        """
        快速制作GIF
        :return:
        """
        if not self.GifInput.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file('请输入要制作成gif的视频文件')
            self.GifInput.setText(input_filename)
            self.GifOutput.setText(
                os.path.join(os.path.dirname(input_filename), f"{Tools.get_filename(input_filename)}.gif"))
            return
        self.function_quick_gif()
        pass

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        """
        快速合并音视频
        :return:
        """
        if not self.ConcatInputV.text():
            self.settings_load_current()  # update settings
            input_filename = self.function_select_file('请输入要进行音视频合并的视频文件')
            self.ConcatInputV.setText(input_filename)
            self.ConcatInputA.setText(input_filename)
            self.OutputConcat.setText(
                os.path.join(os.path.dirname(input_filename), f"{Tools.get_filename(input_filename)}_concat.mp4"))
            return
        self.function_quick_concat()
        pass

    @pyqtSlot(bool)
    def on_ConcatAllButton_clicked(self):
        """
        Only Concat Existed Chunks
        :return:
        """
        self.function_select_first_input()
        self.settings_load_current()  # update settings
        self.ConcatAllButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(concat_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        self.OptionCheck.setText(f"""
                    [SVFI {self.version} 仅合并操作启动，请移步命令行查看进度详情]
                    显示“Program finished”则任务完成
                    如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                    \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_StartExtractButton_clicked(self):
        """
        Only Extract Frames from current input
        :return:
        """
        self.function_select_first_input()
        self.settings_load_current()
        self.StartExtractButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(extract_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        self.OptionCheck.setText(f"""
                            [SVFI {self.version} 仅拆帧操作启动，请移步命令行查看进度详情]
                            显示“Program finished”则任务完成
                            如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                            \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_StartRenderButton_clicked(self):
        """
        Only Render Input based on current settings
        :return:
        """
        self.function_select_first_input()
        self.settings_load_current()
        self.StartRenderButton.setEnabled(False)
        self.tabWidget.setCurrentIndex(1)
        self.progressBar.setValue(0)
        RIFE_thread = SVFI_Run(render_only=True)
        RIFE_thread.run_signal.connect(self.process_update_rife)
        RIFE_thread.start()
        self.rife_thread = RIFE_thread
        self.OptionCheck.setText(f"""
                            [SVFI {self.version} 仅渲染操作启动，请移步命令行查看进度详情]
                            显示“Program finished”则任务完成
                            如果遇到任何问题，请将软件运行界面截图并联系开发人员解决，
                            \n\n\n\n\n""")

    @pyqtSlot(bool)
    def on_KillProcButton_clicked(self):
        """
        Kill Current Process
        :return:
        """
        if self.rife_thread is not None:
            self.rife_thread.kill_proc_exec()

    @pyqtSlot(bool)
    def on_PauseProcess_clicked(self):
        """
        :return:
        """
        if self.rife_thread is not None:
            self.rife_thread.pause_proc_exec()
            if not self.pause:
                self.pause = True
                self.PauseProcess.setText("继续补帧！")
            else:
                self.pause = False
                self.PauseProcess.setText("暂停补帧！")

    def settings_maintain_item_settings(self, widget_data: dict):
        global appData
        self.settings_load_current()  # 保存跳转前设置
        if self.last_item is None:
            self.last_item = widget_data
        config_maintainer = SVFI_Config_Manager(self.last_item, dname)
        config_maintainer.DuplicateConfig()  # 将当前设置保存到上一任务的配置文件，并准备跳转到新任务
        config_maintainer = SVFI_Config_Manager(widget_data, dname)
        config_path = config_maintainer.FetchConfig()
        if config_path is None:
            config_maintainer.DuplicateConfig()  # 利用当前系统全局设置保存当前任务配置
            config_path = config_maintainer.FetchConfig()
        appData = QSettings(config_path, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")
        self.settings_update_pack(True)
        self.last_item = widget_data

    @pyqtSlot(bool)
    def on_ShowAdvance_clicked(self):
        bool_result = self.AdvanceSettingsArea.isVisible()

        # self.settings_maintain_item_settings()

        self.AdvanceSettingsArea.setVisible(not bool_result)
        if not bool_result:
            self.ShowAdvance.setText("隐藏高级设置")
        else:
            self.ShowAdvance.setText("显示高级设置")
        self.splitter.moveSplitter(10000000, 1)

    def SaveInputSettingsProcess(self, current_filename):
        return
        # config_maintainer = SVFI_Config_Manager()
        # config_maintainer.DuplicateConfig(current_filename)

    @pyqtSlot(bool)
    def on_SaveCurrentSettings_clicked(self):
        pass
        # self.settings_load_current()
        # try:
        #     self.SaveInputSettingsProcess(self.InputFileName.currentItem().text())
        #     self.function_send_msg("Success", "当前输入设置保存成功，可直接补帧", 3)
        # except Exception:
        #     self.function_send_msg("Save Failed", "请在输入列表选中要保存设置的条目")
        # pass

    @pyqtSlot(bool)
    def on_LoadCurrentSettings_clicked(self):
        return
        # global appData
        # config_maintainer = SVFI_Config_Manager()
        # try:
        #     current_filename = self.InputFileName.currentItem().text()
        #     if not config_maintainer.MaintainConfig(current_filename):
        #         self.function_send_msg("Not Found Config", "未找到与当前输入匹配的设置文件")
        #         return
        # except Exception:
        #     self.function_send_msg("Load Failed", "请在输入列表选中要加载设置的条目")
        # appData = QSettings(appDataPath, QSettings.IniFormat)
        # appData.setIniCodec("UTF-8")
        # self.settings_initiation()
        # self.function_send_msg("Success", "已载入与当前输入匹配的设置", 3)
        # pass

    @pyqtSlot(bool)
    def on_ClearInputButton_clicked(self):
        self.on_actionClearAllVideos_triggered()

    @pyqtSlot(bool)
    def on_OutputSettingsButton_clicked(self):
        self.function_generate_log(1)
        self.function_send_msg("Generate Settings Log Success", "设置导出成功！settings.log即为设置快照", 3)
        pass

    @pyqtSlot(bool)
    def on_RefreshStartInfo_clicked(self):
        self.settings_set_start_info(-1, -1)
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
            preference_dict["rife_use_cpu"] = self.force_cpu
            preference_dict["is_preview_args"] = self.preview_args
            preference_dict["is_gui_quiet"] = self.is_gui_quiet
            preference_dict["use_clear_inputs"] = self.use_clear_inputs
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
        self.force_cpu = preference_dict["rife_use_cpu"]
        self.preview_args = preference_dict["is_preview_args"]
        self.is_gui_quiet = preference_dict["is_gui_quiet"]
        self.use_clear_inputs = preference_dict["use_clear_inputs"]
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
        self.ScdetOutput.setVisible(self.expert_mode)
        self.ScdetUseMix.setVisible(self.expert_mode)
        self.DeinterlaceChecker.setVisible(self.expert_mode)
        self.FastDenoiseChecker.setVisible(self.expert_mode)
        self.EncodeThreadField.setVisible(self.expert_mode)

        if self.is_free:
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
            currentIndex = self.InputFileName.currentRow()
            self.InputFileName.takeItem(currentIndex)
        except Exception:
            self.function_send_msg("Fail to Clear Video", "未选中输入项")

    @pyqtSlot(bool)
    def on_actionQuit_triggered(self):
        sys.exit(0)

    @pyqtSlot(bool)
    def on_actionClearAllVideos_triggered(self):
        self.InputFileName.clear()
        # self.settings_maintain_item_settings()

    @pyqtSlot(bool)
    def on_actionSaveSettings_triggered(self):
        self.settings_load_current()

    @pyqtSlot(bool)
    def on_actionLoadDefaultSettings_triggered(self):
        appData.clear()
        self.settings_update_pack()
        self.function_send_msg("Load Success", "已载入默认设置", 3)

    def closeEvent(self, event):
        if not self.STEAM.steam_valid:
            event.ignore()
            return
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
