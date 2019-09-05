import logging
import logging.config
import multiprocessing
import os
import shutil
import sqlite3
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QWidget, QAbstractItemView, QTableWidgetItem, QApplication, QMessageBox
from PyQt5.uic import loadUi


# 记录没有找到异常
class RecordNotFound(Exception):
    pass


class DataManageUI(QWidget):
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # 日志信号

    def __init__(self):
        super(DataManageUI, self).__init__()
        # 加载UI
        loadUi('./ui/DataManage.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(931, 577)

        # 设置tableWidget只读，不允许修改
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 数据库
        self.database = './FaceBase.db'  # 数据库地址
        self.datasets = './datasets'  # 数据集地址
        self.isDbReady = False  # 数据库是否已经准备好
        self.initDbButton.clicked.connect(self.initDb)  # 定义初始化数据库按钮点击事件

        # 用户管理
        self.queryUserButton.clicked.connect(self.queryUser)  # 定义查询用户按钮点击事件
        self.deleteUserButton.clicked.connect(self.deleteUser)  # 定义删除用户按钮点击事件

        # 直方图均衡化
        self.isEqualizeHistEnabled = False  # 是否进行直方图均衡化
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.enableEqualizeHist(self.equalizeHistCheckBox)
        )  # 定义直方图均衡化CheckBox点击事件

        # 训练人脸数据,定义开始训练按钮点击按钮事件
        self.trainButton.clicked.connect(self.train)

        # 系统日志
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))  # receiveLogSignal信号绑定事件
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # 定义logOutputThread输出后台线程
        self.logOutputThread.start()  # 开始logOutputThread线程

    # 初始化/刷新数据库,初始化数据库按钮点击事件
    def initDb(self):
        # 刷新前重置tableWidget，清空tableWidget
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
        try:
            if not os.path.isfile(self.database):  # 如果数据库没有找到
                raise FileNotFoundError  # 抛出异常

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            res = cursor.execute('SELECT * FROM users')
            for row_index, row_data in enumerate(res):
                self.tableWidget.insertRow(row_index)
                for col_index, col_data in enumerate(row_data):
                    self.tableWidget.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))  # 显示查询的数据

            cursor.execute('SELECT COUNT(*) FROM users')  # 查询数据库中的样本数
            result = cursor.fetchone()
            dbUserCount = result[0]  # dbUserCount记录样本数
        except FileNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.isDbReady = False
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except Exception:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化/刷新数据库失败')
        else:
            cursor.close()
            conn.close()

            self.dbUserCountLcdNum.display(dbUserCount)
            if not self.isDbReady:
                self.isDbReady = True  # 设置数据库已经准备好
                self.logQueue.put('Success：数据库初始化完成，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setText('刷新数据库')
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                self.trainButton.setToolTip('')
                self.trainButton.setEnabled(True)  # 可以开始训练
                self.queryUserButton.setToolTip('')
                self.queryUserButton.setEnabled(True)
            else:
                self.logQueue.put('Success：刷新数据库成功，发现用户数：{}'.format(dbUserCount))

    # 查询用户，查询用户按钮点击事件
    def queryUser(self):
        stu_id = self.queryUserLineEdit.text().strip()  # 获取学号
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM users WHERE stu_id=?', stu_id)
            ret = cursor.fetchall()
            if not ret:
                raise RecordNotFound
            face_id = ret[0][1]  # 人脸ID
            cn_name = ret[0][2]  # 中文名
        except RecordNotFound:
            self.queryUserButton.setIcon(QIcon('./icons/error.png'))
            self.queryResultLabel.setText('<font color=red>Error：此用户不存在</font>')
        except Exception as e:
            logging.error('读取数据库异常，无法查询到{}的用户信息'.format(stu_id))
            self.queryResultLabel.clear()
            self.queryUserButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，查询失败')
        else:
            self.queryResultLabel.clear()
            self.queryUserButton.setIcon(QIcon('./icons/success.png'))
            # 设置查询到的数据
            self.stuIDLineEdit.setText(stu_id)
            self.cnNameLineEdit.setText(cn_name)
            self.faceIDLineEdit.setText(str(face_id))

            self.deleteUserButton.setEnabled(True)
        finally:
            cursor.close()
            conn.close()

    # 删除用户，删除用户按钮点击事件
    def deleteUser(self):
        text = '从数据库中删除该用户，同时删除相应人脸数据，<font color=red>该操作不可逆！</font>'
        informativeText = '<b>是否继续？</b>'

        ret = DataManageUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
        if ret == QMessageBox.Yes:
            stu_id = self.stuIDLineEdit.text()  # 获得学号
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('DELETE FROM users WHERE stu_id=?', (stu_id,))  # 从数据库中删除记录
            except Exception as e:
                cursor.close()
                logging.error('无法从数据库中删除{}'.format(stu_id))
                self.deleteUserButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，删除失败')
            else:
                cursor.close()
                conn.commit()

                if os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):  # 从本地删除记录
                    try:
                        shutil.rmtree('{}/stu_{}'.format(self.datasets, stu_id))
                    except Exception as e:
                        logging.error('系统无法删除删除{}/stu_{}'.format(self.datasets, stu_id))
                        self.logQueue.put('Error：删除人脸数据失败，请手动删除{}/stu_{}目录'.format(self.datasets, stu_id))

                text = '你已成功删除学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                informativeText = '<b>请在右侧菜单重新训练人脸数据。</b>'
                DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
                # 清空已输入缓存
                self.stuIDLineEdit.clear()
                self.cnNameLineEdit.clear()
                self.faceIDLineEdit.clear()
                self.initDb()
                self.deleteUserButton.setIcon(QIcon('./icons/success.png'))
                self.deleteUserButton.setEnabled(False)
                self.queryUserButton.setIcon(QIcon())
            finally:
                conn.close()

    # 是否执行直方图均衡化,直方图均衡化CheckBox点击事件
    def enableEqualizeHist(self, equalizeHistCheckBox):
        if equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
        else:
            self.isEqualizeHistEnabled = False

    # 检测人脸
    def detectFace(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图，这一步其实可以不用做
        if self.isEqualizeHistEnabled:  # 如果开启了直方图均衡化
            gray = cv2.equalizeHist(gray)  # 进行直方图均衡化
        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')  # 加载opencv官方人脸分类器
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(90, 90))  # 进行人脸检测

        if len(faces) == 0:  #如果没有检测到人脸
            return None, None
        (x, y, w, h) = faces[0]  # 前一步采集的时候保证只有一个人脸
        return gray[y:y + w, x:x + h], faces[0]  # 返回人脸检测区域，和人脸检测信息

    # 准备图片数据
    def prepareTrainingData(self, data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []

        face_id = 1

        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        # 遍历人脸库
        for dir_name in dirs:
            if not dir_name.startswith('stu_'):  # 忽略掉不是以stu开头的文件夹
                continue
            stu_id = dir_name.replace('stu_', '')  # 通过文件夹名获取学号
            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))  # 从数据库中查询
                ret = cursor.fetchall()
                if not ret:  # 如果在数据库中没有找到
                    raise RecordNotFound
                cursor.execute('UPDATE users SET face_id=? WHERE stu_id=?', (face_id, stu_id))  # 再次更新face_id信息
            except RecordNotFound:
                logging.warning('数据库中找不到学号为{}的用户记录'.format(stu_id))
                self.logQueue.put('发现学号为{}的人脸数据，但数据库中找不到相应记录，已忽略'.format(stu_id))
                continue

            subject_dir_path = data_folder_path + '/' + dir_name
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:  # 获取具体用户的人脸图片
                if image_name.startswith('.'):  # 忽略掉以。开头的隐藏文件
                    continue
                image_path = subject_dir_path + '/' + image_name  # 获取图片路径
                image = cv2.imread(image_path)  # 读取图片
                face, rect = self.detectFace(image)  # 调用detectFace检测图片
                if face is not None:  # 如果检测到人脸
                    faces.append(face)
                    labels.append(face_id)
            face_id = face_id + 1

            cursor.close()
            conn.commit()
            conn.close()

            return faces, labels

    # 训练人脸数据,开始训练按钮点击按钮事件
    def train(self):
        try:
            if not os.path.isdir(self.datasets):  # 如果数据集不存在
                raise FileNotFoundError  # 抛出文件不存在异常

            text = '系统将开始训练人脸数据，界面会暂停响应一段时间，完成后会弹出提示。'
            informativeText = '<b>训练过程请勿进行其它操作，是否继续？</b>'
            ret = DataManageUI.callDialog(QMessageBox.Question, text, informativeText,
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)

            if ret == QMessageBox.Yes:
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # 初始化人脸识别器

                if not os.path.exists('./recognizer'):  # 如果不存在trainning数据保存文件夹
                    os.makedirs('./recognizer')  # 创建文件夹

            faces, labels = self.prepareTrainingData(self.datasets)  # 调用prepareTrainingData对数据进行预处理
            face_recognizer.train(faces, np.array(labels))  # 对人脸识别器进行训练
            face_recognizer.save('./recognizer/trainingData.yml')  # 将训练数据保存
        except FileNotFoundError:
            logging.error('系统找不到人脸数据目录{}'.format(self.datasets))
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('未发现人脸数据目录{}，你可能未进行人脸采集'.format(self.datasets))
        except Exception as e:
            logging.error('遍历人脸库出现异常，训练人脸数据失败')
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：遍历人脸库出现异常，训练失败')
        else:
            text = '<font color=green><b>Success!</b></font> 系统已生成./recognizer/trainingData.yml'
            informativeText = '<b>人脸数据训练完成！</b>'
            DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
            self.trainButton.setIcon(QIcon('./icons/success.png'))
            self.logQueue.put('Success：人脸数据训练完成')
            self.initDb()

    # receiveLogSignal信号绑定事件,其中Log为str类型
    def logOutput(self, log):
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

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
        msg.setWindowTitle('OpenCV Face Recognition System - DataManage')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = DataManageUI()
    window.show()
    sys.exit(app.exec())
