# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SVFI_preference.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 540)
        Dialog.setMinimumSize(QtCore.QSize(400, 300))
        Dialog.setMaximumSize(QtCore.QSize(600, 23351))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(300, 300))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 380, 491))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.MultiTaskRestChecker = QtWidgets.QCheckBox(self.groupBox)
        self.MultiTaskRestChecker.setObjectName("MultiTaskRestChecker")
        self.horizontalLayout_2.addWidget(self.MultiTaskRestChecker)
        self.MultiTaskRestInterval = MySpinBox(self.groupBox)
        self.MultiTaskRestInterval.setObjectName("MultiTaskRestInterval")
        self.horizontalLayout_2.addWidget(self.MultiTaskRestInterval)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.AfterMission = QtWidgets.QComboBox(self.groupBox_4)
        self.AfterMission.setObjectName("AfterMission")
        self.AfterMission.addItem("")
        self.AfterMission.addItem("")
        self.AfterMission.addItem("")
        self.gridLayout_5.addWidget(self.AfterMission, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.groupBox_2 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ForceCpuChecker = QtWidgets.QCheckBox(self.groupBox_2)
        self.ForceCpuChecker.setObjectName("ForceCpuChecker")
        self.horizontalLayout_3.addWidget(self.ForceCpuChecker)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout_3.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.ExpertModeChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.ExpertModeChecker.setChecked(True)
        self.ExpertModeChecker.setObjectName("ExpertModeChecker")
        self.gridLayout_4.addWidget(self.ExpertModeChecker, 2, 0, 1, 1)
        self.UseGlobalSettingsChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.UseGlobalSettingsChecker.setObjectName("UseGlobalSettingsChecker")
        self.gridLayout_4.addWidget(self.UseGlobalSettingsChecker, 3, 1, 1, 1)
        self.DisableDedupRenderChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.DisableDedupRenderChecker.setChecked(True)
        self.DisableDedupRenderChecker.setObjectName("DisableDedupRenderChecker")
        self.gridLayout_4.addWidget(self.DisableDedupRenderChecker, 4, 1, 1, 1)
        self.RudeExitModeChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.RudeExitModeChecker.setChecked(True)
        self.RudeExitModeChecker.setObjectName("RudeExitModeChecker")
        self.gridLayout_4.addWidget(self.RudeExitModeChecker, 4, 0, 1, 1)
        self.OneWayModeChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.OneWayModeChecker.setChecked(False)
        self.OneWayModeChecker.setObjectName("OneWayModeChecker")
        self.gridLayout_4.addWidget(self.OneWayModeChecker, 3, 0, 1, 1)
        self.PreviewArgsModeChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.PreviewArgsModeChecker.setChecked(False)
        self.PreviewArgsModeChecker.setObjectName("PreviewArgsModeChecker")
        self.gridLayout_4.addWidget(self.PreviewArgsModeChecker, 2, 1, 1, 1)
        self.PreviewVfiChecker = QtWidgets.QCheckBox(self.groupBox_3)
        self.PreviewVfiChecker.setChecked(True)
        self.PreviewVfiChecker.setObjectName("PreviewVfiChecker")
        self.gridLayout_4.addWidget(self.PreviewVfiChecker, 5, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_3)
        self.groupBox_5 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.QuietModeChecker = QtWidgets.QCheckBox(self.groupBox_5)
        self.QuietModeChecker.setChecked(False)
        self.QuietModeChecker.setObjectName("QuietModeChecker")
        self.gridLayout_6.addWidget(self.QuietModeChecker, 0, 0, 1, 1)
        self.WinOnTopChecker = QtWidgets.QCheckBox(self.groupBox_5)
        self.WinOnTopChecker.setObjectName("WinOnTopChecker")
        self.gridLayout_6.addWidget(self.WinOnTopChecker, 1, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_5)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_3.addItem(spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "多任务休息模式："))
        self.MultiTaskRestChecker.setText(_translate("Dialog", "开启，每隔"))
        self.label_3.setText(_translate("Dialog", "小时休息5分钟"))
        self.AfterMission.setItemText(0, _translate("Dialog", "啥都不做"))
        self.AfterMission.setItemText(1, _translate("Dialog", "关机"))
        self.AfterMission.setItemText(2, _translate("Dialog", "休眠"))
        self.label.setText(_translate("Dialog", "补帧任务完成后："))
        self.ForceCpuChecker.setText(_translate("Dialog", "实验功能：有N卡但强制使用CPU？"))
        self.ExpertModeChecker.setText(_translate("Dialog", "开启专家模式"))
        self.UseGlobalSettingsChecker.setText(_translate("Dialog", "使用全局设置"))
        self.DisableDedupRenderChecker.setText(_translate("Dialog", "无去重压制"))
        self.RudeExitModeChecker.setText(_translate("Dialog", "鲁莽的退出"))
        self.OneWayModeChecker.setText(_translate("Dialog", "任务完成后清空任务列表"))
        self.PreviewArgsModeChecker.setText(_translate("Dialog", "开启任务前参数文本预览"))
        self.PreviewVfiChecker.setText(_translate("Dialog", "开启预览"))
        self.QuietModeChecker.setText(_translate("Dialog", "开启安静模式（不显示弹窗等）"))
        self.WinOnTopChecker.setText(_translate("Dialog", "窗口置顶（下次启动生效）"))
from Utils.RIFE_GUI_Custom import MySpinBox
