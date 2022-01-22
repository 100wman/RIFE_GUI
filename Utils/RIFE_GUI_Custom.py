# -*- coding: utf-8 -*-
import json
import locale
import os
import random
import re
import shutil
import time

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import *

from Utils.StaticParameters import INVALID_CHARACTERS, appDir
from Utils.utils import Tools, ArgumentManager

abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)


class InputItemModel:
    root_config_path = os.path.join(appDir, "SVFI.ini")
    config_dirname = os.path.join(appDir, "Configs")

    def __init__(self, input_path, task_id=None):
        self.input_path = input_path
        self.task_id = task_id
        self.fps = 0
        self.resolution = (0,0)
        self.config_path = ""
        self.is_property_loaded = False
        self.is_config_loaded = False

        if self.task_id is None:
            self.__generate_new_task_id()
        else:
            """check previous config"""
            config_path = self.generate_config_path()
            if os.path.exists(config_path):
                self.initiate_config(config_path)

    def generate_config_path(self):
        config_path = os.path.join(self.config_dirname, f"SVFI_Config_{self.task_id}.ini")
        return config_path

    def __generate_new_task_id(self):
        if not os.path.exists(self.input_path):
            return "TPE_000000"
        path_md5 = Tools.md5(self.input_path)[:3]
        path_id = random.randrange(100000, 999999)
        task_id = f"{path_md5}_{path_id}"
        self.task_id = task_id

    def initiate_property(self, fps=0, resolution=None):
        if not fps and resolution is None:
            self.initiate_input_prop()
        else:
            assert type(resolution) is tuple
            self.resolution = resolution
            self.fps = fps
            self.is_property_loaded = True

    def initiate_config(self, config_path=None):
        if config_path is not None and os.path.exists(config_path):
            """Duplicate from previous config"""
            if not len(self.config_path):  # new model
                self.config_path = config_path
            if config_path != self.config_path:
                shutil.copy(config_path, self.config_path)
        else:
            # Update from root config to generate new task config or template
            assert os.path.exists(self.root_config_path)  # TODO Throw Error here
            self.config_path = self.generate_config_path()
            shutil.copy(self.root_config_path, self.config_path)
        self.is_config_loaded = True

    def initiate_input_prop(self):
        try:
            if not os.path.isfile(self.input_path):
                height, width, fps = 0, 0, 0
            else:
                input_stream = cv2.VideoCapture(self.input_path)
                width = input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = input_stream.get(cv2.CAP_PROP_FPS)
        except Exception:
            height, width, fps = 0, 0, 0
        self.resolution = (int(width), int(height))
        self.fps = fps
        self.is_property_loaded = True

    def get_task_id(self):
        return self.task_id

    def get_input_path(self):
        return self.input_path

    def get_task_info(self):
        return {"input_path": self.input_path, "task_id": self.task_id}

    def get_resolution(self):
        return self.resolution

    def get_fps(self):
        return self.fps

    def get_config_path(self):
        if not os.path.exists(self.config_path):
            return
        return self.config_path

    def update_task_id(self, task_id):
        len_old_id = len(self.task_id)
        self.task_id = task_id
        new_config_path = os.path.splitext(self.config_path)[0][:-len_old_id] + self.task_id + ".ini"
        if os.path.exists(self.config_path):
            if self.config_path != new_config_path:
                shutil.copy(self.config_path, new_config_path)
        else:
            self.initiate_config()
        self.config_path = new_config_path

    def copy(self):
        new_item = InputItemModel(self.input_path)
        new_item.mock(self)
        return new_item

    def mock(self, _InputItemModel):
        self.initiate_property(_InputItemModel.get_fps(), _InputItemModel.get_resolution())
        self.initiate_config(_InputItemModel.get_config_path())

    def apply_config(self):
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, self.root_config_path)

    def delete(self):
        self.config_path = os.path.join(self.config_dirname, f"SVFI_Config_{self.task_id}.ini")
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        pass


class SVFITranslator(QTranslator):
    def __init__(self):
        super().__init__()
        self.app_name = "SVFI"
        try:
            lang = locale.getdefaultlocale()[0].split('_')[0]
            lang_file = self.get_lang_file(lang)
            if not os.path.exists(lang_file):
                """Nor En or Cn, set default to en"""
                lang_file = self.get_lang_file('en')
            self.load(lang_file)
        except Exception as e:
            print(e)

    def get_lang_file(self, lang: str):
        lang_file = os.path.join(dname, 'lang', f'SVFI_UI.{lang}.qm')
        return lang_file

    def change_lang(self, lang: str):
        lang_file = self.get_lang_file(lang)
        self.load(lang_file)


class MyListWidgetItem(QWidget):
    dupSignal = pyqtSignal(InputItemModel)
    remSignal = pyqtSignal(InputItemModel)
    renSignal = pyqtSignal(InputItemModel)

    def __init__(self, parent=None):
        """
        Custom ListWidgetItem to display RIFE Task
        :param parent:
        """
        super().__init__(parent)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.filename = QtWidgets.QLabel(self)
        self.iniCheck = QtWidgets.QCheckBox(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename.sizePolicy().hasHeightForWidth())
        # self.iniCheck.setSizePolicy(sizePolicy)
        # self.ini_checkbox.setMinimumSize(QSize(200, 0))
        self.iniCheck.setObjectName("ini_checkbox")
        self.horizontalLayout.addWidget(self.iniCheck)
        self.filename.setSizePolicy(sizePolicy)
        self.filename.setMinimumSize(QSize(400, 0))
        self.filename.setObjectName("filename")
        self.horizontalLayout.addWidget(self.filename)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.line = QtWidgets.QFrame(self)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.RemoveItemButton = QtWidgets.QPushButton(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RemoveItemButton.sizePolicy().hasHeightForWidth())
        self.task_id_reminder = QtWidgets.QLabel(self)
        self.task_id_reminder.setSizePolicy(sizePolicy)
        self.task_id_reminder.setObjectName("task_id_reminder")
        self.TaskIdDisplay = MyLineWidget(self)
        self.TaskIdDisplay.setSizePolicy(sizePolicy)
        self.TaskIdDisplay.setObjectName("TaskIdDisplay")
        self.horizontalLayout.addWidget(self.task_id_reminder)
        self.horizontalLayout.addWidget(self.TaskIdDisplay)
        self.RemoveItemButton.setSizePolicy(sizePolicy)
        self.RemoveItemButton.setObjectName("RemoveItemButton")
        self.horizontalLayout.addWidget(self.RemoveItemButton)
        self.DuplicateItemButton = QtWidgets.QPushButton(self)
        self.DuplicateItemButton.setSizePolicy(sizePolicy)
        self.DuplicateItemButton.setObjectName("DuplicateItemButton")
        self.horizontalLayout.addWidget(self.DuplicateItemButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.RemoveItemButton.setText("    -    ")
        self.DuplicateItemButton.setText("    +    ")
        # self.iniCheck.setEnabled(False)
        self.RemoveItemButton.clicked.connect(self.on_RemoveItemButton_clicked)
        self.DuplicateItemButton.clicked.connect(self.on_DuplicateItemButton_clicked)
        self.TaskIdDisplay.editingFinished.connect(self.on_TaskIdDisplay_editingFinished)
        """Item Data Settings"""
        self.InputItemModel = None
        self.last_click_time = time.time()

    def setTask(self, _InputItemModel: InputItemModel):
        self.InputItemModel = _InputItemModel
        input_path = self.InputItemModel.get_input_path()
        task_id = self.InputItemModel.get_task_id()
        len_cut = 80
        if len(input_path) > len_cut:
            input_path = os.path.split(input_path)[1]
            if len(input_path) > len_cut:
                input_path = input_path[:len_cut-3] + "..."
            self.filename.setText(input_path[:len_cut])
        else:
            self.filename.setText(input_path)
        self.task_id_reminder.setText("  id:")
        self.TaskIdDisplay.setText(f"{task_id}")

    def get_task_model(self):
        return self.InputItemModel

    def on_DuplicateItemButton_clicked(self, e):
        """
        Duplicate Item Button clicked
        action:
            1: duplicate
            0: remove
        :param e:
        :return:
        """
        self.dupSignal.emit(self.InputItemModel)
        pass

    def on_RemoveItemButton_clicked(self, e):
        self.remSignal.emit(self.InputItemModel)
        pass

    def on_TaskIdDisplay_editingFinished(self):
        self.InputItemModel.update_task_id(self.TaskIdDisplay.text())
        self.renSignal.emit(self.InputItemModel)  # update

    @pyqtSlot(bool)
    def on_iniCheck_toggled(self):
        if time.time() - self.last_click_time > 0.1:
            self.iniCheck.setChecked(not self.iniCheck.isChecked())
            self.last_click_time = time.time()

    def is_checked(self):
        return self.iniCheck.isChecked()


class MyLineWidget(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            url = e.mimeData().urls()[0]
            if not len(self.text()):
                self.setText(url.toLocalFile())
        else:
            e.ignore()


class MyListWidget(QListWidget):
    addSignal = pyqtSignal(int)
    failSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.task_dict = list()

    def checkTaskId(self, task_id):
        """
        Check Task Id Availability
        :param task_id:
        :return: bool
        """
        potential_config_path = os.path.join(InputItemModel.config_dirname, f"SVFI_Config_{task_id}.ini")
        if not os.path.exists(potential_config_path):
            return False
        return True

    def saveTasks(self):
        """
        return tasks information in strings of json format
        :return: {"inputs": [{"task_id": self.task_id, "input_path": self.get_input_path()}]}
        """
        data = list()
        for item in self.getListItems():
            item_model = self.itemWidget(item).get_task_model()
            data.append({"task_id": item_model.get_task_id(), "input_path": item_model.get_input_path()})
        return json.dumps({"inputs": data})

    def dropEvent(self, e):
        if e.mimeData().hasText():  # 是否文本文件格式
            for url in e.mimeData().urls():
                item = url.toLocalFile()
                self.addFileItem(item)
        else:
            e.ignore()

    def dragEnterEvent(self, e):
        self.dropEvent(e)

    def getListItems(self) -> [MyListWidgetItem]:
        """
        获取listwidget中条目数
        :return: list of items
        """
        widgetres = []
        count = self.count()
        # 遍历listwidget中的内容
        for i in range(count):
            widgetres.append(self.item(i))
        return widgetres

    def refreshTasks(self):
        item_models = [self.itemWidget(item).get_task_model() for item in self.getListItems()]
        self.clear()
        try:
            for model in item_models:
                new_item_model, new_item_widget = self.addFileItem(model.get_input_path(), silent=True)  # do not use previous task id
                if new_item_widget is None or new_item_widget is None:
                    continue
                new_item_model.mock(model)
        except RuntimeError:
            pass

    def addConfigItem(self, input_config: str, input_task_id):
        task_id = re.findall("SVFI_Config_(.*?).ini", os.path.basename(input_config))
        if not len(task_id):
            return input_config, input_task_id
        task_id = task_id[0]
        appData = QSettings(input_config, QSettings.IniFormat)
        appData.setIniCodec("UTF-8")
        try:
            input_list_data = json.loads(appData.value("gui_inputs", "{}"))
        except json.decoder.JSONDecodeError:
            return input_config, input_task_id
        for item_data in input_list_data['inputs']:
            if item_data['task_id'] == task_id:
                input_path = item_data['input_path']
                return input_path, task_id
        return None, None

    def addFileItem(self, input_path: str, task_id=None, silent=False):
        """

        :param input_path:
        :param task_id:
        :param silent: avoid activating itemChanged Event in Backend
        :return:
        """
        input_path = input_path.strip('"')
        if len(input_path) > ArgumentManager.path_len_limit:
            self.failSignal.emit(1)  # path too long
            return None, None
        if ArgumentManager.is_free:  # in free version, only one task available
            if self.count() >= 1:
                self.failSignal.emit(2)  # community version does not support multi import
                return None, None
        if any([i in input_path for i in INVALID_CHARACTERS]):
            self.failSignal.emit(3)  # path has invalid character
            return None, None

        if task_id is not None and not self.checkTaskId(task_id):
            return None, None
        taskListItem = MyListWidgetItem()
        _InputItemModel = InputItemModel(input_path, task_id)
        _InputItemModel.initiate_input_prop()
        potential_config_path = None
        if task_id is not None:
            # if exists previous config, use it to initiate
            potential_config_path = _InputItemModel.generate_config_path()
        _InputItemModel.initiate_config(potential_config_path)

        taskListItem.setTask(_InputItemModel)
        taskListItem.dupSignal.connect(self.duplicateItem)
        taskListItem.remSignal.connect(self.removeItem)
        taskListItem.renSignal.connect(self.renameItem)
        # Create QListWidgetItem
        taskListWidgetItem = QListWidgetItem(self)
        # Set size hint
        taskListWidgetItem.setSizeHint(taskListItem.sizeHint())
        # Add QListWidgetItem into QListWidget
        self.addItem(taskListWidgetItem)
        self.setItemWidget(taskListWidgetItem, taskListItem)
        if not silent:
            self.addSignal.emit(self.count())
        return _InputItemModel, taskListWidgetItem

    def duplicateItem(self, _InputItemModel: InputItemModel):
        input_path = _InputItemModel.get_input_path()
        new_item_model, new_item_widget = self.addFileItem(input_path)
        if new_item_widget is None or new_item_widget is None:
            return
        new_item_model.mock(_InputItemModel)
        self.setCurrentItem(new_item_widget)
        return

    def removeItem(self, _InputItemModel: InputItemModel):
        task_id = _InputItemModel.get_task_id()
        for ListItem in self.getListItems():
            possible_item_model = self.itemWidget(ListItem).get_task_model()
            if possible_item_model.get_task_id() == task_id:
                self.takeItem(self.row(ListItem))
        return

    def renameItem(self, _InputItemModel: InputItemModel):
        return

    def keyPressEvent(self, e):
        current_item = self.currentItem()
        if current_item is None:
            e.ignore()
            return
        # if e.key() == Qt.Key_Delete:
        #     self.removeItemWidget(current_item)


class MyTextWidget(QtWidgets.QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dropEvent(self, event):
        try:
            if event.mimeData().hasUrls:
                url = event.mimeData().urls()[0]
                self.setText(f"{url.toLocalFile()}")
            else:
                event.ignore()
        except Exception as e:
            print(e)


class MyComboBox(QComboBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MySpinBox(QSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class MyDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        if e.type() == QEvent.Wheel:
            e.ignore()


class StateTooltip(QWidget):
    """ 进度提示框 """

    def __init__(self, title="", content="", parent=None):
        """
        Parameters
        ----------
        title: str
            状态气泡标题

        content: str
            状态气泡内容

        parant:
            父级窗口
        """
        super().__init__(parent)
        self.title = title
        self.content = content
        # 实例化小部件
        self.titleLabel = QLabel(self.title, self)
        self.contentLabel = QLabel(self.content, self)
        self.rotateTimer = QTimer(self)
        self.closeTimer = QTimer(self)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.busyImage = QPixmap("static/running.png")
        self.doneImage = QPixmap("static/completed.png")
        self.closeButton = QToolButton(self)
        # 初始化参数
        self.isDone = False
        self.rotateAngle = 0
        self.deltaAngle = 20
        # 初始化
        self.__initWidget()
        self.move(QPoint(parent.x() + (parent.width() - self.width()) // 2,
                         parent.y() + (parent.height() - self.height()) // 2))

    def __initWidget(self):
        """ 初始化小部件 """
        self.setAttribute(Qt.WA_StyledBackground)
        self.rotateTimer.setInterval(50)
        self.closeTimer.setInterval(1000)
        self.contentLabel.setMinimumWidth(200)
        # 将信号连接到槽函数
        self.closeButton.clicked.connect(self.hide)  # 点击关闭按钮只是隐藏了提示条
        self.rotateTimer.timeout.connect(self.__rotateTimerFlowSlot)
        self.closeTimer.timeout.connect(self.__slowlyClose)
        self.__setQss()
        self.__initLayout()
        # 打开定时器
        self.rotateTimer.start()

    def __initLayout(self):
        """ 初始化布局 """
        self.titleLabel.adjustSize()
        self.contentLabel.adjustSize()
        self.setFixedSize(max(self.titleLabel.width(),
                              self.contentLabel.width()) + 40, 64)
        self.titleLabel.move(40, 11)
        self.contentLabel.move(15, 34)
        self.closeButton.move(self.width() - 30, 23)

    def __setQss(self):
        """ 设置层叠样式 """
        self.titleLabel.setObjectName("titleLabel")
        self.contentLabel.setObjectName("contentLabel")
        qss = """
        QWidget {
            background-color: #313C4F;
        }
        
        QLabel#titleLabel {
            color: white;
            font: 17px 'Microsoft YaHei';
        }
        
        QLabel#contentLabel {
            color: rgba(255, 255, 255, 210);
            font: 16px 'Microsoft YaHei';
        }
        
        QToolButton{
            border: none;
            margin: 0px;
            width: 14px;
            height: 14px;
            background: url(static/close_normal.png) top center no-repeat;
        }
        
        QToolButton:hover{
            background: url(static/close_hover.png) top center no-repeat;
        }
        
        QToolButton:pressed{
            background: url(static/close_hover.png) top center no-repeat;
        }
        """
        # with open("resource/style/state_tooltip.qss", encoding="utf-8") as f:
        self.setStyleSheet(qss)

    def setTitle(self, title: str):
        """ 设置提示框的标题 """
        self.title = title
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setContent(self, content: str):
        """ 设置提示框内容 """
        self.content = content
        self.contentLabel.setText(content)
        self.contentLabel.adjustSize()

    def setState(self, isDone=False):
        """ 设置运行状态 """
        self.isDone = isDone
        self.update()
        # 运行完成后主动关闭窗口
        if self.isDone:
            self.closeTimer.start()

    def __slowlyClose(self):
        """ 缓慢关闭窗口 """
        self.rotateTimer.stop()
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.setDuration(500)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(self.deleteLater)
        self.animation.start()

    def __rotateTimerFlowSlot(self):
        """ 定时器溢出时旋转箭头 """
        self.rotateAngle = (self.rotateAngle + self.deltaAngle) % 360
        self.update()
        QApplication.processEvents()

    def paintEvent(self, e):
        """ 绘制背景 """
        super().paintEvent(e)
        # 绘制旋转箭头
        painter = QPainter(self)
        painter.setRenderHints(QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        if not self.isDone:
            painter.translate(24, 23)  # 原点平移到旋转中心
            painter.rotate(self.rotateAngle)  # 坐标系旋转
            painter.drawPixmap(
                -int(self.busyImage.width() / 2),
                -int(self.busyImage.height() / 2),
                self.busyImage,
            )
        else:
            painter.drawPixmap(
                14, 13, self.doneImage.width(), self.doneImage.height(), self.doneImage
            )
