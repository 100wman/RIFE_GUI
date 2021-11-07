import sys
import traceback

import win32gui
import win32print
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen
from win32.lib import win32con

import QCandyUi
from Utils.utils import ArgumentManager


version_title = f"Squirrel Video Frame Interpolation {ArgumentManager.version_tag}"

"""SVFI High Resolution Support"""
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    hDC = win32gui.GetDC(0)
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    ArgumentManager.update_screen_size(w, h)
    if w * h >= 3840 * 2160:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

"""Initiate APP"""
app = QApplication(sys.argv)
splash = QSplashScreen(QPixmap("svfi.png"))
splash_qss = """
QSplashScreen {
        font-family: 'Segoe UI';
        font: bold 14px;
        /*font color*/
    }
"""
splash.setStyleSheet(splash_qss)
splash.show()  # 显示启动界面
splash.showMessage("Breaching Graphic Card...", Qt.AlignHCenter | Qt.AlignBottom, Qt.white)
app.processEvents()

import comtypes.client as cc
cc.GetModule("TaskbarLib.tlb")  # note that cometypes __init__ indented two lines on validating the tlb file

import comtypes.gen.TaskbarLib as tbl
taskbar = cc.CreateObject(
    "{56FDF344-FD6D-11d0-958A-006097C9A090}",
    interface=tbl.ITaskbarList3)
taskbar.HrInit()


"""Initiate DLL Env(special bugs here)"""
if ArgumentManager.is_steam:
    try:
        from steamworks import STEAMWORKS  # Import main STEAMWORKS class

        _steamworks = STEAMWORKS(ArgumentManager.app_id)
    except:
        pass

try:
    from Utils import RIFE_GUI_Backend
except ImportError as e:
    exit()

app_backend_module = RIFE_GUI_Backend
app_backend = app_backend_module.UiBackend(splash, taskbar)


try:
    if app_backend.Validation.CheckValidateStart():
        form = QCandyUi.CandyWindow.createWindow(app_backend, theme="blueDeep", ico_path="svfi.png",
                                                 title=version_title)
        app_backend.update_QCandyUi_hwnd(form.winId())
        form.show()
        splash.finish(form)
        app.exec_()
except Exception:
    app_backend_module.logger.critical(traceback.format_exc())
