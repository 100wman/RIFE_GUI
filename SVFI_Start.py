import os
import sys
import QCandyUi
import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QBitmap, QColor, QIcon
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFrame, QLabel, QGraphicsOpacityEffect
import win32gui
import win32con
import PyQt5.Qt, PyQt5.QtCore, PyQt5.QtWidgets
from PyQt5.Qt import QSizePolicy
from PyQt5.QtCore import QEvent, QSize, pyqtSlot
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QApplication, QWidget

# Stretch State
NO_SELECT = 0,  # ���δ�����·���������
LEFT_TOP_RECT = 1  # ��������Ͻ�����
TOP_BORDER = 2  # ������ϱ߿�����
RIGHT_TOP_RECT = 3  # ��������Ͻ�����
RIGHT_BORDER = 4  # ������ұ߿�����
RIGHT_BOTTOM_RECT = 5  # ��������½�����
BOTTOM_BORDER = 6  # ������±߿�����
LEFT_BOTTOM_RECT = 7  # ��������½�����
LEFT_BORDER = 8  # �������߿�����
# ��֪��������
STRETCH_RECT_WIDTH = 4
STRETCH_RECT_HEIGHT = 4

"""High Resolution Support"""
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

try:
    from Utils import SVFI_Backend
except ImportError as e:
    traceback.print_exc()
    print("Not Find RIFE GUI Backend, please contact developers for support")
    input("Press Any Key to Quit")
    exit()

class Titlebar(QWidget):
    # Ĭ�ϻ�����ʽ����
    TITLE_TEXT_COLOR = "white"
    BGD_COLOR = "#28AAAA"
    TITLEBAR_HEIGHT = 30
    ICON_SIZE = QSize(20, 20)
    MIN_BUTT_SIZE = QSize(27, 22)
    MAX_BUTT_SIZE = QSize(27, 22)
    CLOSE_BUTT_SIZE = QSize(27, 22)
    TITLE_LABEL_NAME = "Titlebar_titleLabel"
    BACKGROUND_LABEL_NAME = "Titlebar_backgroundLabel"
    MIN_BUTT_NAME = "Titlebar_minimizeButton"
    MAX_BUTT_NAME = "Titlebar_maximizeButton"
    CLOSE_BUTT_NAME = "Titlebar_closeButton"
    THEME_IMG_DIR = 'default'

    def __init__(self, parent):
        super(Titlebar, self).__init__(parent)
        self.parentwidget = parent
        self.setFixedHeight(Titlebar.TITLEBAR_HEIGHT)
        self.m_pBackgroundLabel = QLabel(self)
        self.m_pIconLabel = QLabel(self)
        self.m_pTitleLabel = QLabel(self)
        self.m_pMinimizeButton = QPushButton(self)
        self.m_pMaximizeButton = QPushButton(self)
        self.m_pCloseButton = QPushButton(self)
        self.m_pIconLabel.setFixedSize(Titlebar.ICON_SIZE)
        self.m_pIconLabel.setScaledContents(True)
        self.m_pTitleLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.m_pBackgroundLabel.setObjectName(Titlebar.BACKGROUND_LABEL_NAME)
        # �����հ�ť��С
        self.m_pMinimizeButton.setFixedSize(Titlebar.MIN_BUTT_SIZE)
        self.m_pMaximizeButton.setFixedSize(Titlebar.MAX_BUTT_SIZE)
        self.m_pCloseButton.setFixedSize(Titlebar.CLOSE_BUTT_SIZE)
        # ͳһ����ObjName
        self.m_pTitleLabel.setObjectName(Titlebar.TITLE_LABEL_NAME)
        self.m_pBackgroundLabel.resize(self.parentwidget.width(), Titlebar.TITLEBAR_HEIGHT)
        self.m_pMinimizeButton.setObjectName(Titlebar.MIN_BUTT_NAME)
        self.m_pMaximizeButton.setObjectName(Titlebar.MAX_BUTT_NAME)
        self.m_pCloseButton.setObjectName(Titlebar.CLOSE_BUTT_NAME)
        # �����հ�ťͼƬ����
        self.setButtonImages()
        # ����
        pLayout = QHBoxLayout(self)
        pLayout.addWidget(self.m_pIconLabel)
        pLayout.addSpacing(5)
        pLayout.addWidget(self.m_pTitleLabel)
        pLayout.addWidget(self.m_pMinimizeButton)
        pLayout.addWidget(self.m_pMaximizeButton)
        pLayout.addWidget(self.m_pCloseButton)
        pLayout.setSpacing(0)
        pLayout.setContentsMargins(5, 0, 5, 0)
        self.setLayout(pLayout)
        # �ź�����
        self.m_pMinimizeButton.clicked.connect(self.__slot_onclicked)
        self.m_pMaximizeButton.clicked.connect(self.__slot_onclicked)
        self.m_pCloseButton.clicked.connect(self.__slot_onclicked)
        # ����Ĭ����ʽ(bar������ɫ�ͱ�����ɫ)
        # self.setTitleBarStyle(Titlebar.BGD_COLOR, Titlebar.TITLE_TEXT_COLOR)

    def setMaximumEnable(self, isenable):
        self.m_pMaximizeButton.setEnabled(isenable)

    def setTitleBarStyle(self, backgroundColor, textColor):
        # ����������ɫ
        self.m_pTitleLabel.setStyleSheet("font-size:13px;margin-bottom:0px;color:%s" % (textColor))
        # ������������ɫ
        self.m_pBackgroundLabel.setStyleSheet("background:%s" % (backgroundColor))

    def setButtonImages(self):
        # �����հ�ťUi����
        # self.m_pCloseButton.setStyleSheet(
        #     self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_CLOSE_NORM, IMG_CLOSE_HOVER, IMG_CLOSE_PRESS, IMG_CLOSE_PRESS))
        # self.m_pMinimizeButton.setStyleSheet(
        #     self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MIN_NORM, IMG_MIN_HOVER, IMG_MIN_PRESS, IMG_MIN_PRESS))
        # self.m_pMaximizeButton.setStyleSheet(
        #     self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MAX_NORM, IMG_MAX_HOVER, IMG_MAX_PRESS, IMG_MAX_PRESS))
        pass

    def __getButtonImgQss(self, root, norm, hover, press, disable):
        qss = str()
        qss += "QPushButton{background:transparent; background-image:url(%s); border:none}" % (
            root + norm)
        qss += "QPushButton:hover{background:transparent; background-image:url(%s)}" % (
            root + hover)
        qss += "QPushButton:pressed{background:transparent; background-image:url(%s)}" % (
            root + press)
        qss += "QPushButton:disabled{background:transparent; background-image:url(%s)}" % (
            root + disable)
        return qss


    def mouseDoubleClickEvent(self, e):
        if self.m_pMaximizeButton.isEnabled():
            self.m_pMaximizeButton.clicked.emit()  # ˫��ȫ��

    def mousePressEvent(self, e):
        """
        ʹ�����ܱ��϶�
        :param e:
        :return:
        """
        win32gui.ReleaseCapture()
        pWindow = self.window()
        if pWindow.isWindow():
            win32gui.SendMessage(pWindow.winId(), win32con.WM_SYSCOMMAND, win32con.SC_MOVE + win32con.HTCAPTION, 0)
        e.ignore()

    def eventFilter(self, object, e):
        if e.type() == QEvent.WindowTitleChange:
            if object != None:
                self.m_pTitleLabel.setText(object.windowTitle())
                return True
        if e.type() == QEvent.WindowIconChange:
            if object != None:
                icon = object.windowIcon()
                self.m_pIconLabel.setPixmap(icon.pixmap(self.m_pIconLabel.size()))
                return True
        if e.type() == QEvent.Resize:
            self.__updateMaxmize()
            self.__setTitleBarSize(self.parentwidget.width())
            return True
        # ע��!����selfҪ����!!!!!!!!!
        return QWidget.eventFilter(self, object, e)

    @pyqtSlot()
    def __slot_onclicked(self):
        pButton = self.sender()
        pWindow = self.window()
        if pWindow.isWindow():
            if pButton.objectName() == Titlebar.MIN_BUTT_NAME:
                pWindow.showMinimized()
            elif pButton.objectName() == Titlebar.MAX_BUTT_NAME:
                if pWindow.isMaximized():
                    pWindow.showNormal()
                    # self.m_pMaximizeButton.setStyleSheet(self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_MAX_NORM, IMG_MAX_HOVER, IMG_MAX_PRESS, IMG_MAX_PRESS))
                else:
                    pWindow.showMaximized()
                    # self.m_pMaximizeButton.setStyleSheet(self.__getButtonImgQss(IMAGE_ROOT + Titlebar.THEME_IMG_DIR + "/", IMG_RESIZE_NORM, IMG_RESIZE_HOVER, IMG_RESIZE_PRESS, IMG_RESIZE_PRESS))
            elif pButton.objectName() == Titlebar.CLOSE_BUTT_NAME:
                pWindow.close()
                os._exit(0)

    def __updateMaxmize(self):
        pWindow = self.window()
        if pWindow.isWindow() == True:
            bMaximize = pWindow.isMaximized()
            if bMaximize == 0:
                self.m_pMaximizeButton.setProperty("maximizeProperty", "restore")
            else:
                self.m_pMaximizeButton.setProperty("maximizeProperty", "maximize")
                self.m_pMaximizeButton.setStyle(QApplication.style())

    def __setTitleBarSize(self, width):
        self.m_pBackgroundLabel.resize(width, Titlebar.TITLEBAR_HEIGHT)


class SVFIWindow(QFrame):
    def __init__(self, mainwidget, parent=0):
        super(SVFIWindow, self).__init__()
        self.setObjectName('WindowWithTitleBar')
        self.m_titlebar = Titlebar(self)
        self.initWidgetsAndPack(mainwidget, self.m_titlebar)
        self.initStretch()
        self.initTipLabel(mainwidget)
        self.setWindowFlags(Qt.FramelessWindowHint)
        # ��������ʽ��ʹ����͸�������� setAttribute(Qt.WA_TranslucentBackground)����Ȼ����Ῠ��
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.95)
        # self.setStyleSheet("background:#CCCCCC")
        # self.windowEffect = WindowEffect()
        # self.windowEffect.setAcrylicEffect(int(self.winId()))
        self.bgColor = QColor(255, 50, 50, 80)

    def initTipLabel(self, parent):
        """
        ��Ϣ��ʾ��ǩ
        :param parent:
        :return:
        """
        self.tipLabel = QLabel(parent)
        self.tipLabel.setFixedSize(200, 40)
        self.tipLabel.setAlignment(Qt.AlignCenter)
        self.tipLabel.setWordWrap(True)
        self.tipLabel.setGeometry(self.width() / 2 - self.tipLabel.width() / 2, self.tipLabel.y(),
                                  self.tipLabel.width(), self.tipLabel.height())
        self.tipLabel.hide()

    def showTip(self, text, color='#20c3ff'):
        if self.tipLabel is not None:
            self.tipLabel.show()
            self.tipLabel.setStyleSheet(
                "QLabel{background: %s;border:5px; color: #FFFFFF; border-radius: 5px}" % color)
            self.tipLabel.setText(text)
            eff = QGraphicsOpacityEffect(self)
            self.tipLabel.setGraphicsEffect(eff)
            self.animate = QPropertyAnimation(eff, "opacity")
            self.animate.setDuration(2000)
            self.animate.setStartValue(0.8)
            self.animate.setEndValue(0)
            self.animate.setEasingCurve(QEasingCurve.InCubic)
            self.animate.finished.connect(lambda: self.tipLabel.hide())
            self.animate.start()

    def initWidgetsAndPack(self, mainwidget, titlebar):
        """
        ������Widget��titleBarƴװ����
        :param mainwidget:
        :param titlebar:
        :return:
        """
        self.mainwidget = mainwidget
        self.resize(mainwidget.width(), mainwidget.height() + Titlebar.TITLEBAR_HEIGHT)
        self.setWindowFlags(Qt.FramelessWindowHint | self.windowFlags())
        self.installEventFilter(titlebar)
        # ����: titlbar��������������
        pLayout = QVBoxLayout(self)
        pLayout.addWidget(titlebar)
        pLayout.addWidget(mainwidget)
        pLayout.setSpacing(0)  # ���еļ���widgetΪ0���
        pLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(pLayout)

    def initStretch(self):
        """
        ��ʼ�����칦��
        :return:
        """
        self.setSupportStretch(True)
        self.m_isWindowMax = False
        self.m_stretchRectState = NO_SELECT
        self.m_isMousePressed = False
        self.setMinimumSize(self.mainwidget.minimumWidth(), self.mainwidget.minimumHeight() + Titlebar.TITLEBAR_HEIGHT)

    def getTitbar(self):
        return self.m_titlebar

    def setMinimumSize(self, width, height):
        """
        �����������СSize
        :param width:
        :param height:
        :return:
        """
        self.m_windowMinWidth = width
        self.m_windowMinHeight = height
        super(SVFIWindow, self).setMinimumSize(width, height)

    def setWindowRadius(self, n_px):
        """
        Բ��
        :param n_px: ����
        :return:
        """
        objBitmap = QBitmap(self.size())
        painter = QPainter(objBitmap)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRoundedRect(self.rect(), n_px, n_px)
        self.setMask(objBitmap)

    def setBackgroundBorderColor(self, bgdcolor, bordercolor):
        self.setStyleSheet("WindowWithTitleBar{background:%s;border:6px solid %s}" % (bgdcolor, bordercolor))

    # def closeEvent(self, *args, **kwargs):
    #     self.mainwidget.close()

    def showEvent(self, event):
        self.calculateCurrentStrechRect()
        return super().showEvent(event)

    def calculateCurrentStrechRect(self):
        # �ĸ���Rect
        self.m_leftTopRect = QRect(0, 0, STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        self.m_leftBottomRect = QRect(0, self.height() - STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH, STRETCH_RECT_WIDTH)
        self.m_rightTopRect = QRect(self.width() - STRETCH_RECT_WIDTH, 0, STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        self.m_rightBottomRect = QRect(self.width() - STRETCH_RECT_WIDTH, self.height() - STRETCH_RECT_HEIGHT,
                                       STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT)
        # ������Rect
        self.m_topBorderRect = QRect(STRETCH_RECT_WIDTH, 0, self.width() - STRETCH_RECT_WIDTH * 2, STRETCH_RECT_HEIGHT)
        self.m_rightBorderRect = QRect(self.width() - STRETCH_RECT_WIDTH, STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH,
                                       self.height() - STRETCH_RECT_HEIGHT * 2)
        self.m_bottomBorderRect = QRect(STRETCH_RECT_WIDTH, self.height() - STRETCH_RECT_HEIGHT,
                                        self.width() - STRETCH_RECT_WIDTH * 2, STRETCH_RECT_HEIGHT)
        self.m_leftBorderRect = QRect(0, STRETCH_RECT_HEIGHT, STRETCH_RECT_WIDTH,
                                      self.height() - STRETCH_RECT_HEIGHT * 2)

    def getCurrentStretchState(self, cursorPos):
        """
        ��������λ�û�ȡStretchState
        :param cursorPos:
        :return:
        """
        if self.m_leftTopRect.contains(cursorPos):
            stretchState = LEFT_TOP_RECT
        elif self.m_rightTopRect.contains(cursorPos):
            stretchState = RIGHT_TOP_RECT
        elif self.m_rightBottomRect.contains(cursorPos):
            stretchState = RIGHT_BOTTOM_RECT
        elif self.m_leftBottomRect.contains(cursorPos):
            stretchState = LEFT_BOTTOM_RECT
        elif self.m_topBorderRect.contains(cursorPos):
            stretchState = TOP_BORDER
        elif self.m_rightBorderRect.contains(cursorPos):
            stretchState = RIGHT_BORDER
        elif self.m_bottomBorderRect.contains(cursorPos):
            stretchState = BOTTOM_BORDER
        elif self.m_leftBorderRect.contains(cursorPos):
            stretchState = LEFT_BORDER
        else:
            stretchState = NO_SELECT
        return stretchState

    def updateMouseStyle(self, stretchState):
        """
        ����stretchStateˢ��������ʽ
        :param stretchState:
        :return:
        """
        if stretchState == NO_SELECT:
            self.setCursor(Qt.ArrowCursor)
        elif stretchState == LEFT_TOP_RECT:
            self.setCursor(Qt.SizeFDiagCursor)
        elif stretchState == RIGHT_BOTTOM_RECT:
            self.setCursor(Qt.SizeFDiagCursor)
        elif stretchState == TOP_BORDER:
            self.setCursor(Qt.SizeVerCursor)
        elif stretchState == BOTTOM_BORDER:
            self.setCursor(Qt.SizeVerCursor)
        elif stretchState == RIGHT_TOP_RECT:
            self.setCursor(Qt.SizeBDiagCursor)
        elif stretchState == LEFT_BOTTOM_RECT:
            self.setCursor(Qt.SizeBDiagCursor)
        elif stretchState == LEFT_BORDER:
            self.setCursor(Qt.SizeHorCursor)
        elif stretchState == RIGHT_BORDER:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """
        ��дmouseMoveEvent�¼������ڻ�ȡ��ǰ����λ�ã���λ�ô��ݸ�getCurrentStretchState������
        �õ���ǰ����״̬��Ȼ�����updateMouseStyle��������ʽ���и���
        :param event:
        :return:
        """
        # �����������ǲ��������
        # Ҳ���ø��������ʽ
        if (self.m_isWindowMax):
            return super().mouseMoveEvent(event)
        # �����ǰ���δ���£�����ݵ�ǰ����λ�ø�������״̬����ʽ
        if not self.m_isMousePressed:
            cursorPos = event.pos()
            # ���ݵ�ǰ����λ����ʾ��ͬ����ʽ
            self.m_stretchRectState = self.getCurrentStretchState(cursorPos)
            self.updateMouseStyle(self.m_stretchRectState)
        # �����ǰ�������Ѿ����£����¼�µڶ������λ�ã������´��ڵĴ�С
        else:
            self.m_endPoint = self.mapToGlobal(event.pos())
            self.updateWindowSize()
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        # ��ǰ������������ָ����8�����򣬲������������ʱ�ſ�ʼ���д�������
        if (self.m_stretchRectState != NO_SELECT and event.button() == Qt.LeftButton):
            self.m_isMousePressed = True
            # ��¼�µ�ǰ���λ�ã�Ϊ�����������λ��
            self.m_startPoint = self.mapToGlobal(event.pos())
            # ����������ǰ�Ĵ���λ�ü���С
            self.m_windowRectBeforeStretch = QRect(self.geometry().x(), self.geometry().y(), self.geometry().width(),
                                                   self.geometry().height())
        # else:
        #     self.windowEffect.moveWindow(self.winId())
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        ����ɿ�����ζ֮��������������ñ�־λ���������¼������������8������Rect
        :param event:
        :return:
        """
        self.m_isMousePressed = False
        self.calculateCurrentStrechRect()
        return super().mouseReleaseEvent(event)

    def updateWindowSize(self):
        """
        ���촰�ڹ����У����ݼ�¼��������´��ڴ�С
        :return:
        """
        windowRect = QRect(self.m_windowRectBeforeStretch.x(), self.m_windowRectBeforeStretch.y(),
                           self.m_windowRectBeforeStretch.width(), self.m_windowRectBeforeStretch.height())
        delValue_X = self.m_startPoint.x() - self.m_endPoint.x()
        delValue_Y = self.m_startPoint.y() - self.m_endPoint.y()
        if self.m_stretchRectState == LEFT_BORDER:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setX(topLeftPoint.x() - delValue_X)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == RIGHT_BORDER:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setX(bottomRightPoint.x() - delValue_X)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == TOP_BORDER:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setY(topLeftPoint.y() - delValue_Y)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == BOTTOM_BORDER:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setY(bottomRightPoint.y() - delValue_Y)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == LEFT_TOP_RECT:
            topLeftPoint = windowRect.topLeft()
            topLeftPoint.setX(topLeftPoint.x() - delValue_X)
            topLeftPoint.setY(topLeftPoint.y() - delValue_Y)
            windowRect.setTopLeft(topLeftPoint)
        elif self.m_stretchRectState == RIGHT_TOP_RECT:
            topRightPoint = windowRect.topRight()
            topRightPoint.setX(topRightPoint.x() - delValue_X)
            topRightPoint.setY(topRightPoint.y() - delValue_Y)
            windowRect.setTopRight(topRightPoint)
        elif self.m_stretchRectState == RIGHT_BOTTOM_RECT:
            bottomRightPoint = windowRect.bottomRight()
            bottomRightPoint.setX(bottomRightPoint.x() - delValue_X)
            bottomRightPoint.setY(bottomRightPoint.y() - delValue_Y)
            windowRect.setBottomRight(bottomRightPoint)
        elif self.m_stretchRectState == LEFT_BOTTOM_RECT:
            bottomLeftPoint = windowRect.bottomLeft()
            bottomLeftPoint.setX(bottomLeftPoint.x() - delValue_X)
            bottomLeftPoint.setY(bottomLeftPoint.y() - delValue_Y)
            windowRect.setBottomLeft(bottomLeftPoint)
        # �������Ϊ�㴰����ʾ�������������������С����߶ȡ����
        if windowRect.width() < self.m_windowMinWidth:
            windowRect.setLeft(self.geometry().left())
            windowRect.setWidth(self.m_windowMinWidth)
        if windowRect.height() < self.m_windowMinHeight:
            windowRect.setTop(self.geometry().top())
            windowRect.setHeight(self.m_windowMinHeight)
        self.setGeometry(windowRect)

    def setSupportStretch(self, isSupportStretch):
        """
        ���õ�ǰ�����Ƿ�֧������
        :param isSupportStretch:
        :return:
        """
        # ��Ϊ��Ҫ�����δ���µ������ͨ��mouseMoveEvent�¼���׽���λ�ã�������Ҫ����setMouseTrackingΪTrue���������֧�����죩
        self.m_isSupportStretch = isSupportStretch
        self.setMouseTracking(isSupportStretch)
        # ������ӿؼ�Ҳ���������ã�����Ϊ��������ӿؼ����ã�������ƶ����ӿؼ���ʱ�����ᷢ��mouseMoveEvent�¼���Ҳ�ͻ�ȡ������ǰ���λ�ã��޷��ж����״̬����ʾ��ʽ�ˡ�
        widgetList = self.findChildren(QWidget)
        for widget in widgetList:
            widget.setMouseTracking(isSupportStretch)
        if (self.m_titlebar is not None):
            # titleBarͬ��,Ҳ��Ҫ���Լ����ӿؼ����е���setMouseTracking�������ã����Ϸ�ע��
            # self.titleBar.setSupportStretch(isSupportStretch)
            pass

    def getSupportStretch(self):
        """
        ���ص�ǰ�����Ƿ�֧������
        :return:
        """
        return self.m_isSupportStretch

    def setMaxEnable(self, isEnable):
        """
        ��󻯿���
        :param isEnable
        """
        self.m_titlebar.setMaximumEnable(isEnable)


app = QApplication(sys.argv)
app_backend_module = SVFI_Backend
app_backend = app_backend_module.RIFE_GUI_BACKEND()

form = SVFIWindow(app_backend)
form.setWindowTitle("Squirrel Video Frame Interpolation 2.3.2 alpha")
form.setWindowIcon(QIcon("svfi.png"))
form.show()
app.exec_()
