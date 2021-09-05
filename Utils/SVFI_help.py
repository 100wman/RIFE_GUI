# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SVFI_help.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(644, 579)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 6, 0, 1, 1)
        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 607, 956))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setScaledContents(True)
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 4, 0, 1, 1)
        self.scrollArea_2 = QtWidgets.QScrollArea(Dialog)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 624, 236))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout.addWidget(self.scrollArea_2, 5, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">快速补帧流程：（鼠标悬浮在选项上可见帮助信息）</span></p><p><span style=\" color:#ffffff;\">1. 选择要补帧的文件。可以选择多个视频文件进行批量补帧，也可以选择包括图片序列的文件夹，后者需要手动填写输入帧率。如果输入的文件有误，</span><span style=\" font-weight:600; color:#ffffff;\">需要重新拖动文件或者点击下方的输入按钮</span></p><p><span style=\" color:#ffffff;\">2. 确认</span><span style=\" font-weight:600; color:#ffffff;\">输入和输出</span><span style=\" color:#ffffff;\">帧率</span></p><p><span style=\" color:#ffffff;\">3. 按“</span><span style=\" font-weight:600; color:#ffffff;\">一键补帧</span><span style=\" color:#ffffff;\">”完成补帧操作，也可以在</span><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">高级设置</span><span style=\" color:#ffffff;\">界面设置相关参数，并在输出界面点击</span><span style=\" font-weight:600; color:#ffffff;\">开始补帧</span><span style=\" color:#ffffff;\">按钮补帧（A卡补帧需要在高级设置手动设置参数）</span></p><p><span style=\" color:#ffffff;\">4. 在</span><span style=\" font-weight:600; color:#ffffff;\">工具箱</span><span style=\" color:#ffffff;\">一页进行</span><span style=\" font-weight:600; color:#ffffff;\">gif制作</span><span style=\" color:#ffffff;\">、</span><span style=\" font-weight:600; color:#ffffff;\">音视频合并</span><span style=\" color:#ffffff;\">及</span><span style=\" font-weight:600; color:#ffffff;\">已有区块合并（视频合并失败使用）</span></p><p><span style=\" font-weight:600; color:#ffffff;\">文件命名缩写说明：</span></p><p><span style=\" color:#ffffff;\">SLM：慢动作输出</span></p><p><span style=\" color:#ffffff;\">DI：DeInterlaced反交错开启</span></p><p><span style=\" color:#ffffff;\">DN：DeNoise降噪开启</span></p><p><span style=\" color:#ffffff;\">NCNN：使用核显或A卡在NCNN架构下补帧</span></p><p><span style=\" color:#ffffff;\">FP16：开启N卡半精度模式</span></p><p><span style=\" color:#ffffff;\">SA：Scale Auto动态光流尺度</span></p><p><span style=\" color:#ffffff;\">RR：RIFE-Reversed反向光流</span></p><p><span style=\" color:#ffffff;\">RFE：RIFE-Foward Ensemble双向光流</span></p><p><span style=\" color:#ffffff;\">TTA：TTA模式</span></p><p><span style=\" color:#ffffff;\">SR：SuperResolution使用超分</span></p><p><span style=\" color:#ffffff;\">RD：Remove Duplicates去除重复帧模式</span></p><p><span style=\" font-weight:600; color:#ffffff;\">注意事项：</span></p><p><span style=\" color:#ffffff;\">显示“Programfinished”则任务完成</span></p><p><span style=\" color:#ffffff;\">如果遇到任何问题，请将基础设置、输出窗口截全图，不要录屏，并导出当前设置为settings.log文件并联系开发人员解决，群号在首页说明</span></p><p><span style=\" color:#ffffff;\"><br/></span></p><p><span style=\" color:#ffffff;\">参数说明：</span></p><p><span style=\" color:#ffffff;\">R:当前渲染的帧数，</span></p><p><span style=\" color:#ffffff;\">C:当前处理的帧数，</span></p><p><span style=\" color:#ffffff;\">S:最近识别到的转场，</span></p><p><span style=\" color:#ffffff;\">SC：识别到的转场数量，</span></p><p><span style=\" color:#ffffff;\">TAT：TaskAcquireTime，单帧任务获取时间，即任务阻塞时间，如果该值过大，请考虑增加虚拟内存</span></p><p><span style=\" color:#ffffff;\">PT：ProcessTime，单帧任务处理时间，单帧补帧（+超分）花费时间</span></p><p><span style=\" color:#ffffff;\">QL：QueueLength，任务队列长度</span></p><p><span style=\" color:#ffffff;\"><br/></span></p><p><span style=\" color:#ffffff;\">如果遇到卡顿或软件卡死，请直接右上角强制终止</span></p><p><span style=\" font-weight:600; font-style:italic; color:#ffffff;\">软件最终解释权归SVFI开发团队所有</span></p><p><span style=\" color:#ffffff;\"><br/></span></p></body></html>"))
        self.label.setText(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600; color:#ffffff;\">SVFI 3.5.16 更新日志</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#ffffff;\">    - Fix Queue Len to optimize Dedup Mode (Fix 200 as minimum queue len for 300px img)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#ffffff;\">    - Optimize swscale</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#ffffff;\">    - Update QSVEnCc Release Version -&gt; 6.0.0</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#ffffff;\">    - Add shell, process: start, execute, make a copy, rewind, close, rename, release</span></p></body></html>"))
