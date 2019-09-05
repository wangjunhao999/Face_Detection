import logging
import logging.config
import multiprocessing
import os
import queue
import sqlite3
import sys
import threading
import webbrowser
import winsound
from configparser import ConfigParser
from datetime import datetime

import cv2
import dlib
import telegram
from PyQt5.QtCore import pyqtSignal, QThread, QTimer, Qt, QRegExp
from PyQt5.QtGui import QIcon, QImage, QPixmap, QRegExpValidator, QTextCursor
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QDialog
from PyQt5.uic import loadUi
from PyQt5.uic.properties import QtGui


class TrainingDataNotFoundError(FileNotFoundError):  # 训练数据没有找到错误
    pass


class DataBaseNotFoundError(FileNotFoundError):  # 数据库没有找到错误
    pass


# 人脸检测线程
class FaceProcessingThread(QThread):
    def __init__(self):
        super(FaceProcessingThread, self).__init__()
        self.isRunning = True  # 线程是否正在运行

        self.isFaceTrackerEnabled = True  # 是否允许进行人脸跟踪
        self.isFaceRecognizerEnabled = False  # 是否允许进行人脸识别
        self.isPanalarmEnabled = True  # 是否允许进行报警

        self.isDebugMode = False  # 是否处于Debug模式
        self.confidenceThreshold = 50  # 置信度阈值
        self.autoAlarmThreshold = 65  # 自动报警阈值

        self.isEqualizeHistEnabled = False  # 是否允许进行直方图均衡化

    # 是否开启人脸跟踪，人脸跟踪CheckBox点击事件
    def enableFaceTracker(self, coreUI):
        if coreUI.faceTrackerCheckBox.isChecked():
            self.isFaceTrackerEnabled = True
            coreUI.statusBar().showMessage('人脸跟踪：开启')
        else:
            self.isFaceTrackerEnabled = False
            coreUI.statusBar().showMessage('人脸跟踪：关闭')

    # 是否开启人脸识别,人脸识别CheckBox点击事件
    def enableFaceRecognizer(self, coreUI):
        if coreUI.faceRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isFaceRecognizerEnabled = True
                coreUI.statusBar().showMessage('人脸识别：开启')
            else:
                CoreUI.logQueue.put('Error：操作失败，请先开启人脸跟踪')
                coreUI.faceRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.faceRecognizerCheckBox.setChecked(False)
        else:
            self.isFaceRecognizerEnabled = False
            coreUI.statusBar().showMessage('人脸识别：关闭')

    # 是否开启报警系统,报警系统CheckBox点击事件
    def enablePanalarm(self, coreUI):
        if coreUI.panalarmCheckBox.isChecked():
            self.isPanalarmEnabled = True
            coreUI.statusBar().showMessage('报警系统：开启')
        else:
            self.isPanalarmEnabled = False
            coreUI.statusBar().showMessage('报警系统：关闭')

    # 是否开启调试模式，调试模式CheckBox点击事件
    def enableDebug(self, coreUI):
        if coreUI.debugCheckBox.isChecked():
            self.isDebugMode = True
            coreUI.statusBar().showMessage('调试模式：开启')
        else:
            self.isDebugMode = False
            coreUI.statusBar().showMessage('调试模式：关闭')

    # 设置置信度阈值,置信度阈值滑动事件
    def setConfidenceThreshold(self, coreUI):
        if self.isDebugMode:
            self.confidenceThreshold = coreUI.confidenceThresholdSlider.value()
            coreUI.statusBar().showMessage('置信度阈值：{}'.format(self.confidenceThreshold))

    # 设置自动报警阈值，自动报警阈值滑动事件
    def setAutoAlarmThreshold(self, coreUI):
        if self.isDebugMode:
            self.autoAlarmThreshold = coreUI.autoAlarmThresholdSlider.value()
            coreUI.statusBar().showMessage('自动报警阈值：{}'.format(self.autoAlarmThreshold))

    # 直方图均衡化,直方图均衡化按钮点击事件
    def enableEqualizeHist(self, coreUI):
        if coreUI.equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
            coreUI.statusBar().showMessage('直方图均衡化：开启')
        else:
            self.isEqualizeHistEnabled = False
            coreUI.statusBar().showMessage('直方图均衡化：关闭')

    def run(self):
        # 加载OpenCV官方人脸分类器
        faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        # 帧数,人脸ID初始化
        frameCounter = 0  # 帧数
        currentFaceID = 0  # 当前人脸ID

        # 人脸跟踪器字典初始化,每个键值对均为一个人脸跟踪器
        faceTrackers = {}

        isTrainingDataLoaded = False  # 训练数据是否已经加载
        isDbConnected = False  # 数据库是否连接成功

        while self.isRunning:  # 当程序正在运行
            if CoreUI.cap.isOpened():  # 如果相机已经打开
                ret, frame = CoreUI.cap.read()  # 尝试读取一张图片
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图片转换为灰度图

                # 是否进行直方图均衡化
                if self.isEqualizeHistEnabled:  # 如果允许进行直方图均衡化
                    gray = cv2.equalizeHist(gray)  # 进行直方图均衡化

                faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))  # 检测人脸

                # 预加载数据文件
                if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):
                    recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建人脸分类器
                    recognizer.read(CoreUI.trainingData)  # 加载已经读取好的数据模型
                    isTrainingDataLoaded = True  # 训练数据模型读取完毕

                # 连接数据库
                if not isDbConnected and os.path.isfile(CoreUI.database):
                    conn = sqlite3.connect(CoreUI.database)
                    cursor = conn.cursor()
                    isDbConnected = True

                captureData = {}  # 照片数据
                realTimeFrame = frame.copy()  # 真实图片
                alarmSignal = {}  # 报警信号

                # 人脸跟踪
                if self.isFaceTrackerEnabled:  # 如果允许进行人脸跟踪
                    # 要删除的人脸跟踪器列表初始化
                    fidsToDelete = []  # 这里指定要删除的人脸分类器

                    for fid in faceTrackers.keys():  # 对于每个人脸分类器
                        # 实时跟踪
                        trackingQuality = faceTrackers[fid].update(realTimeFrame)  # 对于每个分类器重新进行评分
                        # 如果跟踪质量过低,则删除该人脸跟踪器
                        if trackingQuality < 7:
                            fidsToDelete.append(fid)

                    # 删除跟踪质量过低的人脸跟踪器
                    for fid in fidsToDelete:
                        faceTrackers.pop(fid, None)

                    for _x, _y, _w, _h in faces:  # 对于OpenCV检测到的人脸
                        isKnown = False  # 默认是陌生人

                        if self.isFaceRecognizerEnabled:  # 如果允许进行人脸识别
                            cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (2323, 138, 30), 2)  # 绘制人脸区域
                            face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])  # 对人脸进行预测获得人脸ID和置信度
                            logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                            if self.isDebugMode:  # 如果处于debug模式
                                CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                            # 从数据库中获取识别人脸的身份信息

                            try:
                                cursor.execute('SELECT * FROM users WHERE face_id=?', (face_id,))  # 尝试在数据库中获取该人脸信息
                                result = cursor.fetchall()
                                if result:
                                    en_name = result[0][3]  # 获取该face_id对应的英文名
                                else:
                                    raise Exception
                            except Exception as e:
                                logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                en_name = ''

                            # 若置信度评分小于置信度阈值，认为是可靠识别
                            if confidence < self.confidenceThreshold:
                                isKnown = True  # 该身份在数据库中已存在
                                cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                            1,
                                            (0, 97, 255), 2)  # 绘制该人员身份英文名
                            else:  # 不可靠识别
                                # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                                cv2.putText(realTimeFrame, 'unknown', (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                                # 若置信度评分超出自动报警阈值，触发报警信号
                                if confidence > self.autoAlarmThreshold:  # 大于自动报警阈值
                                    if self.isPanalarmEnabled:  # 如果允许进行报警
                                        alarmSignal['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')
                                        alarmSignal['img'] = realTimeFrame
                                        CoreUI.alarmQueue.put(alarmSignal)
                                        logging.info('系统发出了报警信号')

                        # 帧数自增
                        frameCounter += 1

                        # 每读取10帧，检测跟踪器的人脸是否还在当前画面内
                        if frameCounter % 10 == 0:
                            # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                            # 而dlib人脸跟踪器要求的是int类型
                            x = int(_x)
                            y = int(_y)
                            w = int(_w)
                            h = int(_h)

                            # 计算中心点
                            x_bar = x + 0.5 * w
                            y_bar = y + 0.5 * h

                            # matchedFid表征当前检测到的人脸是否已被跟踪
                            matchedFid = None

                            for fid in faceTrackers.keys():
                                # 获取人脸跟踪器的位置
                                # tracked_position是dlib.drectangle类型,用来表征图像的矩形区域,坐标是浮点数
                                tracked_position = faceTrackers[fid].get_position()

                                # 浮点数取整
                                t_x = int(tracked_position.left())
                                t_y = int(tracked_position.top())
                                t_w = int(tracked_position.width())
                                t_h = int(tracked_position.height())

                                # 计算人脸跟踪器的中心点
                                t_x_bar = t_x + 0.5 * t_w
                                t_y_bar = t_y + 0.5 * t_h

                                # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                                # 说明当前人脸已被跟踪
                                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                        (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                    matchedFid = fid

                            # 如果当前检测到的人脸是陌生人脸且未被跟踪
                            if not isKnown and matchedFid is None:
                                # 创建一个人脸跟踪器
                                tracker = dlib.correlation_tracker()
                                # 锁定跟踪范围
                                tracker.start_track(realTimeFrame, dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                                # 将该人脸跟踪器分配给当前检测到的人脸
                                faceTrackers[currentFaceID] = tracker
                                # 人脸ID自增
                                currentFaceID += 1

                    # 使用当前的人脸跟踪器，更新画面，输出跟踪结果
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # 在跟踪帧中圈出人脸
                        cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                        cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                                    2)

                captureData['originFrame'] = frame
                captureData['realTimeFrame'] = realTimeFrame
                CoreUI.captureQueue.put(captureData)

            else:
                continue

    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()


class TelegramBotDialog(QDialog):
    def __init__(self):
        super(TelegramBotDialog, self).__init__()
        # 加载UI
        loadUi('./ui/TelegramBotDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(550, 358)

        chat_id_regx = QRegExp('^\d+$')  # 设置ID只能为数字
        chat_id_validator = QRegExpValidator(chat_id_regx, self.telegramIDLineEdit)
        self.telegramIDLineEdit.setValidator(chat_id_validator)

        self.okButton.clicked.connect(self.telegramBotSettings)  # 设置更新配置按钮点击事件

    # 更新配置按钮点击事件
    def telegramBotSettings(self):
        # 获取用户输入
        token = self.tokenLineEdit.text().strip()
        chat_id = self.telegramIDLineEdit.text().strip()
        proxy_url = self.socksLineEdit.text().strip()
        message = self.messagePlainTextEdit.toPlainText().strip()

        # 校验并处理用户输入
        if not (token and chat_id and message):  # 如果输入不满足格式
            self.okButton.setIcon(QIcon('./icons/error.png'))
            CoreUI.logQueue.put('Error：API Token、Telegram ID和消息内容为必填项')
        else:
            ret = self.telegramBotTest(token, proxy_url)
            if ret:  # 如果连接成功
                cfg_file = './config/telegramBot.cfg'
                cfg = ConfigParser()
                cfg.read(cfg_file, encoding='utf-8-sig')

                # 尝试写入配置
                cfg.set('telegramBot', 'token', token)
                cfg.set('telegramBot', 'chat_id', chat_id)
                cfg.set('telegramBot', 'proxy_url', proxy_url)
                cfg.set('telegramBot', 'message', message)

                try:
                    with open(cfg_file, 'w', encoding='utf-8') as file:
                        cfg.write(file)
                except:
                    logging.error('写入telegramBot配置文件发生异常')
                    CoreUI.logQueue.put('Error：写入配置文件时发生异常，更新失败')
                else:
                    CoreUI.logQueue.put('Success：测试通过，系统已更新TelegramBot配置')
                    self.close()
            else:
                CoreUI.logQueue.put('Error：测试失败，无法更新TelegramBot配置')

    # TelegramBot 测试是否连接成功
    def telegramBotTest(self, token, proxy_url):
        try:
            # 是否使用代理
            if proxy_url:
                proxy = telegram.utils.request.Request(proxy_url=proxy_url)
                bot = telegram.Bot(token=token, request=proxy)
            else:
                bot = telegram.Bot(token=token)
            bot.get_me()
        except Exception as e:
            return False
        else:
            return True


# CoreUI实现类
class CoreUI(QMainWindow):
    database = './FaceBase.db'  # 数据库位置
    trainingData = './recognizer/trainingData.yml'  # 训练数据模型位置

    cap = cv2.VideoCapture()  # OpenCV
    captureQueue = queue.Queue()  # 图像队列
    alarmQueue = queue.LifoQueue()  # 报警队列，后进先出
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # log信号

    def __init__(self):
        super(CoreUI, self).__init__()
        # 加载UI
        loadUi('./ui/Core.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(1161, 623)

        # 图像捕获
        self.isExternalCameraUsed = False  # 是否使用外部摄像头
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox)
        )  # 设置使用外接摄像头CheckBox按钮事件

        self.faceProcessingThread = FaceProcessingThread()  # 人脸检测线程
        self.startWebcamButton.clicked.connect(self.startWebcam)  # 设置打开摄像头按钮事件

        # 数据库
        self.initDbButton.setIcon(QIcon('./icons/warning.png'))
        self.initDbButton.clicked.connect(self.initDb)  # 设置检查数据库状态按钮事件

        self.timer = QTimer(self)  # 设置定时器
        self.timer.timeout.connect(self.updateFrame)  # 设置定时器触发事件

        # 功能开关
        # 设置人脸跟踪CheckBox点击事件
        self.faceTrackerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceTracker(self))
        # 设置人脸识别CheckBox点击事件
        self.faceRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceRecognizer(self))
        # 设置报警系统CheckBox点击事件
        self.panalarmCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enablePanalarm(self))

        # 设置直方图均衡化按钮点击事件
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableEqualizeHist(self))

        # 调试模式
        # 设置调试模式CheckBox点击事件
        self.debugCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableDebug(self))
        # 设置置信度阈值滑动事件
        self.confidenceThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setConfidenceThreshold(self))
        # 设置自动报警阈值滑动事件
        self.autoAlarmThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setAutoAlarmThreshold(self))

        # 报警系统
        self.alarmSignalThreshold = 10  # 设置报警信号数量阈值
        self.panalarmThread = threading.Thread(target=self.recievieAlarm, daemon=True)  # 设置自动报警线程

        # 个性化设置
        self.isBellEnabled = True  # 设备发声允许
        self.bellCheckBox.stateChanged.connect(
            lambda: self.enableBell(self.bellCheckBox))  # 设置设备发声CheckBox点击事件

        self.isTelegramBotPushEnabled = False  # TelegramBot推送禁止
        self.telegramBotPushCheckBox.stateChanged.connect(
            lambda: self.enableTelegramBotPush(self.telegramBotPushCheckBox))  # 设置TelegramBot CheckBox点击事件
        self.telegramBotSettingsButton.clicked.connect(self.telegramBotSettings)  # 设置TelegramBot设置按钮点击事件

        # 帮助与支持
        self.viewGithubRepoButton.clicked.connect(
            lambda: webbrowser.open('https://github.com/wangjunhao999/Face_Detection'))  # 设置Github仓库按钮点击事件
        self.contactDeveloperButton.clicked.connect(lambda: webbrowser.open('http://www.nicomoe.cn'))  # 设置联系开发者按钮点击事件

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))  # 绑定receiveLogSignal信号到logOutput处理函数
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # 定义日志后台打印线程
        self.logOutputThread.start()  # 启动日志后台打印线程

    # 使用外接摄像头CheckBox按钮事件
    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    # 打开摄像头按钮事件
    def startWebcam(self):
        if not self.cap.isOpened():  # 如果相机没有打开
            if self.isExternalCameraUsed:  # 如果是使用的外部摄像头
                camID = 1  # 相机ID = 1
            else:  # 使用笔记本自带的摄像头
                camID = 0  # 相机ID = 0

            self.cap.open(camID)  # 打开摄像头
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置Frame宽度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置Frame高度
            ret, frame = self.cap.read()  # 预读取一张图片

            if not ret:  # 如果没有读取成功
                logging.error("无法调用电脑摄像头{}".format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()  # 释放摄像头
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
            else:  # 摄像头正常，读取成功
                self.faceProcessingThread.start()  # 启动OpenCV人脸检测线程
                self.timer.start(5)  # 启动定时器
                self.panalarmThread.start()  # 启动报警器线程
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
                self.startWebcamButton.setText('关闭摄像头')

        else:  # 摄像头已经打开

            text = '如果关闭摄像头，须重启程序才能再次打开。'
            informativeText = '<b>是否继续？</b>'
            ret = CoreUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

            if ret == QMessageBox.Yes:  # 用户选择继续关闭摄像头
                self.faceProcessingThread.stop()  # 停止人脸检测线程
                if self.cap.isOpened():  # 如果摄像头已打开
                    if self.timer.isActive():  # 如果定时器在启动
                        self.timer.stop()  # 关闭定时器
                    self.cap.release()  # 释放摄像头

                self.realTimeCaptureLabel.clear()  # 清理数据
                self.realTimeCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('摄像头已关闭')
                self.startWebcamButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    # 查数据库状态按钮事件
    def initDb(self):
        try:
            if not os.path.isfile(self.database):  # 如果数据库文件不存在
                raise DataBaseNotFoundError  # 抛出数据库没有找到错误
            if not os.path.isfile(self.trainingData):    # 如果训练数据模型不存在
                raise TrainingDataNotFoundError  # 抛出训练数据模型没有找到异常

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM users')  # 在数据库中查询用户样本数
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DataBaseNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error: 未发现数据库文件，你可能未进行人脸采集')
        except TrainingDataNotFoundError:
            logging.error('系统找不到已训练的人脸数据{}'.format(self.trainingData))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现已训练的人脸数据文件，请完成训练后继续')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:  # 如果数据库没有样本
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
                self.initDbButton.setIcon(QIcon('./icons/warning.png'))
            else:
                self.logQueue.put('Success：数据库状态正常，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                self.initDbButton.setEnabled(False)
                self.faceRecognizerCheckBox.setToolTip('须先开启人脸跟踪')
                self.faceRecognizerCheckBox.setEnabled(True)

    # 定时器触发事件，展示图片
    def updateFrame(self):
        if self.cap.isOpened(): # 确保摄像头已经打开
            if not self.captureQueue.empty():  # 图像队列不为空
                captureData = self.captureQueue.get()  # 获取图片数据
                realTimeFrame = captureData.get('realTimeFrame')  # 获得实时图片
                self.displayImage(realTimeFrame, self.realTimeCaptureLabel)  # 展示图片

    #展示图片，updateFrame调用子程序
    def displayImage(self, img, qlabel):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #将图片转换为RGB格式
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8  # 默认为8为灰度图片

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:  # 如果图片是RGB+透明度彩色图片
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:  # 否则是RGB彩色图片
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))  # 展示图片
        qlabel.setScaledContents(True)  # 图片自适应大小

    # 报警系统服务常驻，接收并处理报警信号，自动报警线程触发事件
    def recievieAlarm(self):
        while True:
            jobs = []
            # 大于设定报警信号阈值
            if self.alarmQueue.qsize() > self.alarmSignalThreshold:  # 如果报警队列大于设定的最低报警数量阈值
                if not os.path.isdir('./unknown'):  # 如果不存在unknown文件夹
                    os.makedirs('./unknown')  # 创建unknown文件夹
                lastAlarmSignal = self.alarmQueue.get()  # 获取最近的报警信号
                timestamp = lastAlarmSignal.get('timestamp')  # 获取报警信号时间戳
                img = lastAlarmSignal.get('img')  # 获取报警信号图片
                # 疑似陌生人脸，截屏存档
                cv2.imwrite('./unknown/{}.jpg'.format(timestamp), img)  # 将陌生人脸保存到unknown文件夹
                logging.info('报警信号触发超出预设计数，自动报警系统已被激活')
                self.logQueue.put('Info：报警信号触发超出预设计数，自动报警系统已被激活')

                # 是否进行响铃
                if self.isBellEnabled:  # 如果可以进行响铃
                    p1 = multiprocessing.Process(target=CoreUI.bellProcess, args=(self.logQueue,))  # 定义设备响铃进程
                    p1.start()  # 启动进程
                    jobs.append(p1)   # 将进程保存到jobs里面进行管理

                # 是否进行TelegramBot推送
                if self.isTelegramBotPushEnabled:
                    if os.path.isfile('./unknown/{}.jpg'.format(timestamp)):
                        img = './unknown/{}.jpg'.format(timestamp)
                    else:
                        img = None
                    p2 = multiprocessing.Process(target=CoreUI.telegramBotPushProcess, args=(self.logQueue, img))  # 定义TelegramBot推送进程
                    p2.start()  # 启动TelegramBot推送进程
                    jobs.append(p2)  # 将进程保存到jobs里面进行管理

                # 等待本轮报警结束
                for p in jobs:
                    p.join()

                # 重置报警信号
                with self.alarmQueue.mutex:
                    self.alarmQueue.queue.clear()

            # 小于设定报警信号阈值
            else:
                continue

    # 报警系统：是否允许设备响铃,设备发声CheckBox点击事件
    def enableBell(self, bellCheckBox):
        # 报警
        if bellCheckBox.isChecked():
            self.isBellEnabled = True
            self.statusBar().showMessage("设备发声：开启")
        else:  # 没有勾选报警
            if self.isTelegramBotPushEnabled:
                self.isBellEnabled = False
                self.statusBar().showMessage("设备发声：关闭")
            else:
                self.logQueue.put('Error：操作失败，至少选择一种报警方式')
                self.bellCheckBox.setCheckState(Qt.Unchecked)
                self.bellCheckBox.setChecked(True)

    # 报警系统：是否允许TelegramBot推送,TelegramBot CheckBox点击事件
    def enableTelegramBotPush(self, telegramBotPushCheckBox):
        if telegramBotPushCheckBox.isChecked():
            self.isTelegramBotPushEnabled = True
            self.statusBar().showMessage('TelegramBot推送：开启')
        else:
            if self.isBellEnabled:
                self.isTelegramBotPushEnabled = False
                self.statusBar().showMessage('TelegramBot推送：关闭')
            else:
                self.logQueue.put('Error：操作失败，至少选择一种报警方式')
                self.telegramBotPushCheckBox.setCheckState(Qt.Unchecked)
                self.telegramBotPushCheckBox.setChecked(True)

    # TelegramBot设置按钮点击事件
    def telegramBotSettings(self):
        cfg = ConfigParser()    # 定义一个配置类
        cfg.read('./config/telegramBot.cfg', encoding='utf-8-sig')  # 读取配置
        read_only = cfg.getboolean('telegramBot', 'read_only')  # 获取telegramBot组下read_only配置

        if read_only:  # 如果只允许读取
            text = '基于安全考虑，系统拒绝了本次请求。'
            informativeText = '<b>请联系设备管理员。</b>'
            CoreUI.callDialog(QMessageBox.Critical, text, informativeText, QMessageBox.Ok)  # 显示不允许进行配置
        else:
            token = cfg.get('telegramBot', 'token')  # 获取token信息
            chat_id = cfg.get('telegramBot', 'chat_id')  # 获取chat_id信息
            proxy_url = cfg.get('telegramBot', 'proxy_url')  # 获取proxy_url信息
            message = cfg.get('telegramBot', 'message')  # 获取message信息

            self.telegramBotDialog = TelegramBotDialog()  # 定义TelegramBotDialog实例
            self.telegramBotDialog.tokenLineEdit.setText(token)  # 设置配置信息
            self.telegramBotDialog.telegramIDLineEdit.setText(chat_id)
            self.telegramBotDialog.socksLineEdit.setText(proxy_url)
            self.telegramBotDialog.messagePlainTextEdit.setPlainText(message)
            self.telegramBotDialog.exec()  # 启动TelegramBotDialog实例

    # 设备响铃进程
    @staticmethod
    def bellProcess(queue):
        logQueue = queue
        logQueue.put('Info：设备正在响铃...')
        winsound.PlaySound('./alarm.wav', winsound.SND_FILENAME)

    # TelegramBot推送进程
    @staticmethod
    def telegramBotPushProcess(queue, img=None):
        logQueue = queue
        cfg = ConfigParser()
        try:
            cfg.read('./config/telegramBot.cfg', encoding='utf-8-sig')

            # 读取TelegramBot配置
            token = cfg.get('telegramBot', 'token')
            chat_id = cfg.getint('telegramBot', 'chat_id')
            proxy_url = cfg.get('telegramBot', 'proxy_url')
            message = cfg.get('telegramBot', 'message')

            # 是否使用代理
            if proxy_url:  # 如果使用代理
                proxy = telegram.utils.request.Request(proxy_url=proxy_url)
                bot = telegram.Bot(token=token, request=proxy)
            else:
                bot = telegram.Bot(token=token)

            bot.send_message(chat_id=chat_id, text=message)  # 发送message给chat_id用户

            # 发送疑似陌生人脸截屏到Telegram
            if img:  # 如果有陌生人图片
                bot.send_photo(chat_id=chat_id, photo=open(img, 'rb'), timeout=10)  # 发送photo给chat_id用户

        except Exception as e:
            logQueue.put('Error：TelegramBot推送失败')
        else:
            logQueue.put('Success：TelegramBot推送成功')

    # LOG输出，receiveLogSignal信号处理函数
    def logOutput(self, log):
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - Core')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    def closeEvent(self, event):
        if self.faceProcessingThread.isRunning:
            self.faceProcessingThread.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = CoreUI()
    window.show()
    sys.exit(app.exec_())
