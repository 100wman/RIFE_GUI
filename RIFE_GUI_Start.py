import os
import sys
import traceback

import win32gui
import win32print
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from win32.lib import win32con

import QCandyUi
from Utils.utils import ArgumentManager

version_title = f"Squirrel Video Frame Interpolation {ArgumentManager.version_tag}"

if ArgumentManager.is_steam:
    """Initiate DLL Env(special bugs here)"""
    original_cwd = os.getcwd()
    try:
        from steamworks import STEAMWORKS  # Import main STEAMWORKS class

        steamworks = STEAMWORKS(ArgumentManager.app_id)
        steamworks.initialize()  # This method has to be called in order for the wrapper to become functional!
    except:
        pass
    os.chdir(original_cwd)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

try:
    from Utils import RIFE_GUI_Backend
except ImportError as e:
    traceback.print_exc()
    print("Not Find RIFE GUI Backend, please contact developers for support")
    input("Press Any Key to Quit")
    exit()

"""SVFI High Resolution Support"""
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    hDC = win32gui.GetDC(0)
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    if w * h >= 3840 * 2160:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

"""Initiate APP"""
app = QApplication(sys.argv)
app_backend_module = RIFE_GUI_Backend
app_backend = app_backend_module.RIFE_GUI_BACKEND()
try:
    if app_backend.STEAM.steam_valid:
        form = QCandyUi.CandyWindow.createWindow(app_backend, theme="blueDeep", ico_path="svfi.png",
                                                 title=version_title)
        form.show()
        app.exec_()
except Exception:
    app_backend_module.logger.critical(traceback.format_exc())
