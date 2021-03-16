import sys
import QCandyUi
from PyQt5.QtWidgets import *

try:
    from Utils import RIFE_GUI_Backend
except ImportError as e:
    print("Not Find RIFE GUI Backend, please contact developers for support")
    input("Press Any Key to Quit")
    exit()
app = QApplication(sys.argv)
form = QCandyUi.CandyWindow.createWindow(RIFE_GUI_Backend.RIFE_GUI_BACKEND(), theme="blueDeep", ico_path="icon.ico",
                                         title="Squirrel Video Frame Interpolation Ft. RIFE GUI v6.2.1")
# TODO Change Version to SVFI
form.show()
app.exec_()
