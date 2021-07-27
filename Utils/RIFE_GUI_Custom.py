# -*- coding: utf-8 -*-
import json
import os
import random
import shutil

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Utils.utils import Tools

abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))
ddname = os.path.dirname(abspath)

class MyListWidgetItem(QWidget):
    dupSignal = pyqtSignal(dict)
    remSignal = pyqtSignal(dict)

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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename.sizePolicy().hasHeightForWidth())
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
        self.RemoveItemButton.clicked.connect(self.on_RemoveItemButton_clicked)
        self.DuplicateItemButton.clicked.connect(self.on_DuplicateItemButton_clicked)
        self.TaskIdDisplay.editingFinished.connect(self.on_TaskIdDisplay_editingFinished)
        """Item Data Settings"""
        self.task_id = None
        self.input_path = None

    def setTask(self, input_path: str, task_id: str):
        self.task_id = task_id
        self.input_path = input_path
        len_cut = 100
        if len(self.input_path) > len_cut:
            self.filename.setText(self.input_path[:len_cut] + "...")
        else:
            self.filename.setText(self.input_path)
        self.task_id_reminder.setText("  id:")
        self.TaskIdDisplay.setText(f"{self.task_id}")

    def get_task_info(self):
        self.task_id = self.TaskIdDisplay.text()
        return {"task_id": self.task_id, "input_path": self.input_path}

    def on_DuplicateItemButton_clicked(self, e):
        """
        Duplicate Item Button clicked
        action:
            1: duplicate
            0: remove
        :param e:
        :return:
        """
        emit_data = self.get_task_info()
        emit_data.update({"action": 1})
        self.dupSignal.emit(emit_data)
        pass

    def on_RemoveItemButton_clicked(self, e):
        emit_data = self.get_task_info()
        emit_data.update({"action": 0})
        self.dupSignal.emit(emit_data)
        pass

    def on_TaskIdDisplay_editingFinished(self):
        previous_task_id = self.task_id
        emit_data = self.get_task_info()  # update task id by the way
        emit_data.update({"previous_task_id": previous_task_id, "action": 3})
        self.dupSignal.emit(emit_data)  # update

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.task_dict = list()
        # self.setDragDropMode(QAbstractItemView.InternalMove)

    def generateTaskId(self, input_path: str):
        path_md5 = Tools.md5(input_path)[:6]
        while True:
            path_id = random.randrange(100000, 999999)
            if path_id not in self.task_dict:
                self.task_dict.append(path_id)
                break
        task_id = f"{path_md5}_{path_id}"
        return task_id

    def saveTasks(self):
        """
        return tasks information in strings of json format
        :return: {"inputs": [{"task_id": self.task_id, "input_path": self.input_path}]}
        """
        data = list()
        for item in self.getItems():
            widget = self.itemWidget(item)
            item_data = widget.get_task_info()
            data.append(item_data)
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

    def getWidgetData(self, item):
        """
        Get widget data from item's widget
        :param item: item
        :return: dict of widget data, including row
        """
        try:
            widget = self.itemWidget(item)
            item_data = widget.get_task_info()
            item_data.update({"row": self.row(item)})
        except AttributeError:
            return None
        return item_data

    def getItems(self):
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
        items = [self.getWidgetData(item) for item in self.getItems()]
        self.clear()
        try:
            for item in items:
                new_item_data = self.addFileItem(item['input_path'], item['task_id'])
                config_maintainer = SVFI_Config_Manager(new_item_data, dname)
                config_maintainer.DuplicateConfig(item)  # use previous item data to establish config file

        except RuntimeError:
            pass

    def addFileItem(self, input_path: str, task_id=None) -> dict:
        input_path = input_path.strip('"')
        taskListItem = MyListWidgetItem()
        if task_id is None:
            task_id = self.generateTaskId(input_path)
        taskListItem.setTask(input_path, task_id)
        taskListItem.dupSignal.connect(self.itemActionResponse)
        taskListItem.remSignal.connect(self.itemActionResponse)
        # Create QListWidgetItem
        taskListWidgetItem = QListWidgetItem(self)
        # Set size hint
        taskListWidgetItem.setSizeHint(taskListItem.sizeHint())
        # Add QListWidgetItem into QListWidget
        self.addItem(taskListWidgetItem)
        self.setItemWidget(taskListWidgetItem, taskListItem)
        item_data = {"input_path": input_path, "task_id": task_id}
        self.addSignal.emit(self.count())
        return item_data

    def itemActionResponse(self, e: dict):
        """
        Respond to item's action(click on buttons)
        :param e:
        :return:
        """
        """
        self.dupSignal.emit({"task_id": self.task_id, "input_path": self.input_path, "action": 1})
        """
        task_id = e.get('task_id')
        target_item = None
        for item in self.getItems():
            task_data = self.itemWidget(item).get_task_info()
            if task_data['task_id'] == task_id:
                target_item = item
                break
        if target_item is None:
            return
        if e.get("action") == 1:  # dupSignal
            input_path = self.itemWidget(target_item).input_path
            new_item_data = self.addFileItem(input_path)
            config_maintainer = SVFI_Config_Manager(new_item_data, dname)
            config_maintainer.DuplicateConfig(self.getWidgetData(target_item))
            pass
        elif e.get("action") == 0:
            self.takeItem(self.row(target_item))
        elif e.get("action") == 3:
            item_data = self.getWidgetData(target_item)
            config_maintainer = SVFI_Config_Manager(item_data, dname)
            previous_item_data = {"input_path": item_data['input_path'], "task_id": e.get("previous_task_id")}
            config_maintainer.DuplicateConfig(previous_item_data)

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


class SVFI_Config_Manager:
    """
    SVFI 配置文件管理类
    """

    def __init__(self, item_data: dict, app_dir: str, _logger=None):

        self.input_path = item_data['input_path']
        self.task_id = item_data['task_id']
        self.dirname = os.path.join(app_dir, "Configs")
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        self.SVFI_config_path = os.path.join(app_dir, "SVFI.ini")
        self.config_path = self.__generate_config_path()
        if _logger is None:
            self.logger = Tools.get_logger("ConfigManager","")
        else:
            self.logger = _logger
        pass

    def FetchConfig(self):
        """
        根据输入文件名获得配置文件路径
        :return:
        """
        if os.path.exists(self.config_path):
            return self.config_path
        else:
            return None

    def DuplicateConfig(self, item_data=None):
        """
        复制配置文件
        :return:
        """
        if not os.path.exists(self.SVFI_config_path):
            self.logger.warning("Not find Basic Config")
            return False
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if item_data is not None:
            """Duplicate from previous item_data"""
            previous_config_manager = SVFI_Config_Manager(item_data, dname)
            previous_config_path = previous_config_manager.FetchConfig()
            if previous_config_path is not None:
                """Previous Item Data is not None"""
                shutil.copy(previous_config_path, self.config_path)
        else:
            shutil.copy(self.SVFI_config_path, self.config_path)
        return True

    def RemoveConfig(self):
        """
        移除配置文件
        :return:
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        else:
            self.logger.warning("Not find Config to remove, guess executed directly from main file")
        pass

    def MaintainConfig(self):
        """
        维护配置文件,在LoadSettings后维护
        :return:
        """
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, self.SVFI_config_path)
            return True
        else:
            return False
        pass

    def __generate_config_path(self):
        return os.path.join(self.dirname, f"SVFI_Config_{self.task_id}.ini")