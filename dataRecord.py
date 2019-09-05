# 用户取消了更新数据库操作
import logging.config
import os
import queue
import sqlite3
import sys
import threading
from datetime import datetime

import cv2
from PyQt5.QtCore import pyqtSignal, QTimer, QRegExp
from PyQt5.QtGui import QIcon, QImage, QPixmap, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QDialog
from PyQt5.uic import loadUi


# 用户取消了更新数据库操作
class OperationCancel(Exception):
    pass


# 采集过程中出现干扰
class RecordDisturbance(Exception):
    pass


# 用户信息填写对话框
class UserInfoDialog(QDialog):
    def __init__(self):
        super(UserInfoDialog, self).__init__()
        # 初始化UI
        loadUi('./ui/UserInfoDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(425, 300)

        # 使用正则表达式限制用户输入
        # 1.限制用户学号输入
        stu_id_regx = QRegExp('^[0-9]{12}$')
        stu_id_validator = QRegExpValidator(stu_id_regx, self.stuIDLineEdit)
        self.stuIDLineEdit.setValidator(stu_id_validator)
        # 2.限制用户姓名输入
        cn_name_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        cn_name_validator = QRegExpValidator(cn_name_regx, self.cnNameLineEdit)
        self.cnNameLineEdit.setValidator(cn_name_validator)
        # 3.限制用户汉语拼音输入
        en_name_regx = QRegExp('^[ A-Za-z]{1,16}$')
        en_name_validator = QRegExpValidator(en_name_regx, self.enNameLineEdit)
        self.enNameLineEdit.setValidator(en_name_validator)


class DataRecordUI(QWidget):
    # 传递Log信号
    receiveLogSignal = pyqtSignal(str)

    def __init__(self):
        super(DataRecordUI, self).__init__()
        # 初始化UI
        loadUi('./ui/DataRecord.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(1011, 601)

        # OpenCV
        self.cap = cv2.VideoCapture()
        self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        self.logQueue = queue.Queue()  # 日志队列

        # 图像捕获
        self.isExternalCameraUsed = False  # 是否使用外接摄像头
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox)
        )  # 定义使用外接摄像头CheckBox点击事件
        self.startWebcamButton.toggled.connect(self.startWebcam)  # 定义打开摄像头按钮点击实践
        self.startWebcamButton.setCheckable(True)

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)  # 利用定时器更新Frame

        # 人脸检测
        self.isFaceDetectEnabled = False
        self.enableFaceDetectButton.toggled.connect(self.enableFaceDetect)  # 定义开启人脸检测按钮点击事件
        self.enableFaceDetectButton.setCheckable(True)

        # 数据库
        self.database = './FaceBase.db'  # 数据库地址
        self.datasets = './datasets'  # 数据集地址
        self.isDbReady = False  # 这个状态表示数据库是否准备好
        self.initDbButton.setIcon(QIcon('./icons/warning.png'))
        self.initDbButton.clicked.connect(self.initDb)  # 设置初始化数据库按钮点击事件

        # 用户信息
        self.isUserInfoReady = False  # 这个状态表示用户信息是否已经准备好
        self.userInfo = {'stu_id': '', 'cn_name': '', 'en_name': ''}  # 这里存储输入的用户信息
        self.addOrUpdateUserInfoButton.clicked.connect(self.addOrUpdateUserInfo)  # 设置增加用户/修改用户资料按钮点击事件
        self.migrateToDbButton.clicked.connect(self.migrateToDb)  # 设置同步到数据库点击按钮事件

        # 人脸采集
        self.startFaceRecordButton.clicked.connect(
            lambda: self.startFaceRecord(self.startFaceRecordButton)
        )  # 定义开始采集人脸数据按钮点击事件
        self.faceRecordCount = 0  # 已经采集的人脸数量
        self.minFaceRecordCount = 100  # 最低需要采集的人脸数量
        self.isFaceDataReady = False  # 人脸数据是否准备好
        self.isFaceRecordEnabled = False  # 人脸采集是否允许
        self.enableFaceRecordButton.clicked.connect(self.enableFaceRecord)  # 采集当前捕获帧按钮点击事件

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))  # receiveLogSignal信号绑定事件
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # Log输入后台线程
        self.logOutputThread.start()  # 启动Log线程

    # 是否使用外接摄像头CheckBox点击事件
    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    # 打开/关闭摄像头按钮点击事件
    def startWebcam(self, status):
        if status:  # 打开摄像头
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)  # 打开摄像头
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置Frame宽度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置Frame高度
            ret, frame = self.cap.read()  # 预读取一张Frame

            if not ret:  # 如果读取失败
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()  # 释放当前摄像头
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
                self.startWebcamButton.setChecked(False)
            else:  # 如果读取成功
                self.startWebcamButton.setText('关闭摄像头')
                self.enableFaceDetectButton.setEnabled(True)  # 允许点击开启人脸检测按钮
                self.timer.start(5)  # 利用定时器更新Frame
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
        else:  # 关闭摄像头
            if self.cap.isOpened():
                if self.timer.isActive():
                    self.timer.stop()  # 停止Frame更新
                self.cap.release()  # 释放当前摄像头
                self.faceDetectCaptureLabel.clear()
                self.faceDetectCaptureLabel.setText('<font color=red>摄像头未开启</font>')  # 设置label显示为摄像头未开启
                self.startWebcamButton.setText('打开摄像头')  # 修改为打开摄像头
                self.enableFaceDetectButton.setEnabled(False)  # 禁止点击人脸检测按钮
                self.startWebcamButton.setIcon(QIcon())

    # 开启/关闭人脸检测按钮点击事件
    def enableFaceDetect(self, status):
        if self.cap.isOpened():  # 摄像头必须打开
            if status:  # 开启人脸检测
                self.enableFaceDetectButton.setText('关闭人脸检测')
                self.isFaceDetectEnabled = True
            else:  # 关闭人脸检测
                self.enableFaceDetectButton.setText('开启人脸检测')
                self.isFaceDetectEnabled = False

    # 采集当前捕获帧按钮点击事件
    def enableFaceRecord(self):
        if not self.isFaceRecordEnabled:
            self.isFaceRecordEnabled = True

    # timer定时器事件，不断更新Frame
    def updateFrame(self):
        ret, frame = self.cap.read()  # 读取一帧
        # self.image = cv2.flip(self.image, 1)
        if ret:  # 读取成功
            self.displayImage(frame)  # 展示图片

            if self.isFaceDetectEnabled:  # 如果开启了人脸检测
                detected_frame = self.detectFace(frame)
                self.displayImage(detected_frame)  # 展示检测人脸后的图片
            else:
                self.displayImage(frame)  # 直接展示原来的图片

    # 初始化数据库按钮点击事件
    def initDb(self):
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()
        try:
            # 检测人脸数据目录是否存在，不存在则创建
            if not os.path.isdir(self.datasets):
                os.makedirs(self.datasets)

            # 查询数据表是否存在，不存在则创建
            # 学号，人脸编号，中文名，英文名，创建日期
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              stu_id VARCHAR(12) PRIMARY KEY NOT NULL,
                              face_id INTEGER DEFAULT -1,
                              cn_name VARCHAR(10) NOT NULL,
                              en_name VARCHAR(16) NOT NULL,
                              created_time DATE DEFAULT (date('now','localtime'))
                              )
                          ''')
            # 查询数据表记录数
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]  # 记录当前用户数量
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False  # 数据库没有准备好
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：初始化数据库失败')
        else:
            self.isDbReady = True  # 数据库已经准备好
            self.dbUserCountLcdNum.display(dbUserCount)  # 显示数据库已存人脸样本数
            self.logQueue.put('Success：数据库初始化完成')
            self.initDbButton.setIcon(QIcon('./icons/success.png'))
            self.initDbButton.setEnabled(False)
            self.addOrUpdateUserInfoButton.setEnabled(True)  # 允许点击添加或修改用户信息按钮
        finally:
            cursor.close()
            conn.commit()
            conn.close()

    # 在图片中检测人脸，updateFrame程序调用
    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将Frame转换为灰度图
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))  # 加载OPENCV官方的人脸分类器

        stu_id = self.userInfo.get('stu_id')  # 获得当前用户学号

        for (x, y, w, h) in faces:
            if self.isFaceRecordEnabled:  # 允许进行人脸采集
                try:
                    if not os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):  # 如果不存在该学生datasets
                        # ，则创建当前学生datasets
                        os.makedirs('{}/stu_{}'.format(self.datasets, stu_id))
                    if len(faces) > 1:  # 采集到多张人脸
                        raise RecordDisturbance
                    # 保存已采集图片
                    cv2.imwrite('{}/stu_{}/img.{}.jpg'.format(self.datasets, stu_id, self.faceRecordCount + 1),
                                gray[y - 20:y + h + 20, x - 20:x + w + 20])
                except RecordDisturbance:  # 捕获到采集到多张人脸干扰异常
                    self.isFaceRecordEnabled = False
                    logging.error('检测到多张人脸或环境干扰')
                    self.logQueue.put('Warning：检测到多张人脸或环境干扰，请解决问题后继续')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/warning.png'))
                    continue
                except Exception as e:  # 捕获到其他异常
                    logging.error('写入人脸图像文件到计算机过程中发生异常')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.logQueue.put('Error：无法保存人脸图像，采集当前捕获帧失败')
                else:  # 没有出现异常
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.faceRecordCount = self.faceRecordCount + 1  # 增加已采集人脸样本数
                    self.isFaceRecordEnabled = False
                    self.faceRecordCountLcdNum.display(self.faceRecordCount)
            cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 2)  # 在Frame上绘制人脸矩形

        return frame  # 返回绘制矩形后的Frame

    # 增加用户/修改用户资料按钮点击事件
    def addOrUpdateUserInfo(self):
        self.userInfoDialog = UserInfoDialog()

        stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
            'en_name')  # 尝试获取上次添加或修改的用户信息
        self.userInfoDialog.stuIDLineEdit.setText(stu_id)
        self.userInfoDialog.cnNameLineEdit.setText(cn_name)
        self.userInfoDialog.enNameLineEdit.setText(en_name)

        self.userInfoDialog.okButton.clicked.connect(self.checkToApplyUserInfo)  # 填好了按钮点击事件
        self.userInfoDialog.exec()  # 显示userInfoDialog

    # 校验用户信息并提交，填好了按钮点击事件
    def checkToApplyUserInfo(self):
        if not (self.userInfoDialog.stuIDLineEdit.hasAcceptableInput() and
                self.userInfoDialog.cnNameLineEdit.hasAcceptableInput() and
                self.userInfoDialog.enNameLineEdit.hasAcceptableInput()):  # 如果输入有误
            self.userInfoDialog.msgLabel.setText('<font color=red>你的输入有误，提交失败，请检查并重试！</font>')
        else:  # 如果输入符合格式
            # 获取用户输入，保存到userInfo字典里面
            self.userInfo['stu_id'] = self.userInfoDialog.stuIDLineEdit.text().strip()
            self.userInfo['cn_name'] = self.userInfoDialog.cnNameLineEdit.text().strip()
            self.userInfo['en_name'] = self.userInfoDialog.enNameLineEdit.text().strip()
            # 信息确认
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            self.stuIDLineEdit.setText(stu_id)  # 在当前UI中显示输入的学号
            self.cnNameLineEdit.setText(cn_name)  # 在当前UI中显示输入的中文名
            self.enNameLineEdit.setText(en_name)  # 在当前UI中显示输入的英文名
            self.isUserInfoReady = True  # 用户信息已经输入完毕
            if not self.startFaceRecordButton.isEnabled():  # 可以开始采集
                self.startFaceRecordButton.setEnabled(True)
            self.migrateToDbButton.setIcon(QIcon())
            # 关闭对话框
            self.userInfoDialog.close()

    # 同步到数据库点击按钮事件
    def migrateToDb(self):
        if self.isFaceDataReady:  # 人脸数据已经尊卑好
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')  # 获取学号，中文名，英文名
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                if cursor.fetchall():  # 如果已经存在该用户
                    text = '数据库已存在学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                    informativeText = '<b>是否覆盖？</b>'
                    ret = DataRecordUI.callDialog(QMessageBox.Warning, text, informativeText,
                                                  QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        # 更新已有记录
                        cursor.execute('UPDATE users SET cn_name=?, en_name=? WHERE stu_id=?',
                                       (cn_name, en_name, stu_id,))
                    else:
                        raise OperationCancel  # 记录取消覆盖操作
                else:  # 数据库中不存在该用户
                    # 插入新记录
                    cursor.execute('INSERT INTO users (stu_id, cn_name, en_name) VALUES (?, ?, ?)',
                                   (stu_id, cn_name, en_name,))

                cursor.execute('SELECT Count(*) FROM users')
                result = cursor.fetchone()
                dbUserCount = result[0]  # 更新dbUserCount信息
            except OperationCancel:
                pass
            except Exception as e:
                logging.error('读写数据库异常，无法向数据库插入/更新记录')
                self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，同步失败')
            else:
                text = '<font color=blue>{}</font> 已添加/更新到数据库。'.format(stu_id)
                informativeText = '<b><font color=blue>{}</font> 的人脸数据采集已完成！</b>'.format(cn_name)
                DataRecordUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                # 清空用户信息缓存
                for key in self.userInfo.keys():
                    self.userInfo[key] = ''
                self.isUserInfoReady = False

                self.faceRecordCount = 0
                self.isFaceDataReady = False
                self.faceRecordCountLcdNum.display(self.faceRecordCount)
                self.dbUserCountLcdNum.display(dbUserCount)

                # 清空历史输入
                self.stuIDLineEdit.clear()
                self.cnNameLineEdit.clear()
                self.enNameLineEdit.clear()
                self.migrateToDbButton.setIcon(QIcon('./icons/success.png'))

                # 允许继续增加新用户
                self.addOrUpdateUserInfoButton.setEnabled(True)
                self.migrateToDbButton.setEnabled(False)

            finally:
                cursor.close()
                conn.commit()
                conn.close()
        else:
            self.logQueue.put('Error：操作失败，你尚未完成人脸数据采集')
            self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))

    # 开始采集人脸数据按钮点击事件
    def startFaceRecord(self, startFaceRecordButton):
        if startFaceRecordButton.text() == '开始采集人脸数据':
            if self.isFaceDetectEnabled:  # 允许人脸检测
                if self.isUserInfoReady:  # 用户信息已经准备好
                    self.addOrUpdateUserInfoButton.setEnabled(False)  # 此时禁止添加或者修改用户信息
                    if not self.enableFaceRecordButton.isEnabled():  # 允许点击采集当前捕获帧按钮
                        self.enableFaceRecordButton.setEnabled(True)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.startFaceRecordButton.setText('结束当前人脸采集')
                else:  # 用户信息没有准备好
                    self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.startFaceRecordButton.setChecked(False)
                    self.logQueue.put('Error：操作失败，系统未检测到有效的用户信息')
            else:  # 不允许进行人脸检测
                self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：操作失败，请开启人脸检测')
        else:  # 点击采集结束
            if self.faceRecordCount < self.minFaceRecordCount:  # 采集数量小于当前最低要求的人脸采集数量
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，采集数据过少会导致较大的识别误差。'.format(self.faceRecordCount)
                informativeText = '<b>请至少采集 <font color=red>{}</font> 帧图像。</b>'.format(self.minFaceRecordCount)
                DataRecordUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

            else: # 采集数量大于当前最低要求的人脸采集数量
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，继续采集可以提高识别准确率。'.format(self.faceRecordCount)
                informativeText = '<b>你确定结束当前人脸采集吗？</b>'
                ret = DataRecordUI.callDialog(QMessageBox.Question, text, informativeText,
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)

                if ret == QMessageBox.Yes:
                    self.isFaceDataReady = True  # 人脸数据已经准备好
                    if self.isFaceRecordEnabled:  # 禁止人脸数据采集
                        self.isFaceRecordEnabled = False
                    self.enableFaceRecordButton.setEnabled(False)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setText('开始采集人脸数据')
                    self.startFaceRecordButton.setEnabled(False)
                    self.startFaceRecordButton.setIcon(QIcon())
                    self.migrateToDbButton.setEnabled(True)

    # receiveLogSignal信号绑定事件，输出LOG日志，log参数为str类型
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    # 获得log并发出receiveLogSignal信号，从而logOutput捕获到进行输出
    def receiveLog(self):
        while True:
            data = self.logQueue.get()  # 获取log队列中的log
            if data:
                self.receiveLogSignal.emit(data)  # 发出receiveLogSignal信号
            else:
                continue

    # 显示图像，updateFrame程序调用
    def displayImage(self, img):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8  #默认为灰度图片

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4: # 4通道，即RGB+透明度彩色图片
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:  # 3通道，RGB彩色图片
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.faceDetectCaptureLabel.setPixmap(QPixmap.fromImage(outImage))  # 设置显示图片
        self.faceDetectCaptureLabel.setScaledContents(True)  # 设置自适应

    # 系统对话框
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - DataRecord')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    # 窗口关闭事件，关闭定时器、摄像头
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = DataRecordUI()
    # window = UserInfoDialog()
    window.show()
    sys.exit(app.exec())
