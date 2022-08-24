from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, \
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QTabWidget
from PyQt5.QtGui import QPixmap, QImage, QStaticText
import mediapipe as mp
import numpy as np
import time, os, math, threading, cv2, sys
from playsound import playsound


class poseDetection():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionConf, self.trackConf)

    # 포즈 찾기
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPostion(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 150), cv2.FILLED)

        return lmList


class Thread1(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        global running
        cap = cv2.VideoCapture(0)
        while running:
            ret, img = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1669, 1233, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class Thread2(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        global running2
        cap = cv2.VideoCapture(0)
        while running2:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1669, 1233, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class Thread_AMSR(QThread):

    def run(self):
        playsound('mp3file/hello_school.mp3')


class Thread_Heart(QThread):

    def run(self):
        playsound('mp3file/blue.mp3')


class Thread_Left(QThread):

    def run(self):
        playsound('mp3file/left.mp3')


class Thread_Right(QThread):

    def run(self):
        playsound('mp3file/right.mp3')


# class scale():
#     def __init__(self, p):
#         super().__init__()
#         self.p = p
#
#     def pp(self):
#         global p
#         self.p = p


class Thread3(QThread):
    changePixmap = pyqtSignal(QImage)
    changeText = pyqtSignal(str)

    def run(self):
        global heart, arms, left, right, cap, running3
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('image_2/1 (2).mp4')
        # cap = cv2.VideoCapture('data_video/IMG_0327.MOV')
        print('a')
        #cap = cv2.VideoCapture('data_video/FINALLY/Heart.MOV')
        pTime = 0
        detector = poseDetection()
        l = []
        j = []
        while running3:
            ret, img = cap.read()
            if ret:
                img = detector.findPose(img)
                lmList = detector.findPostion(img, draw=False)
                # 11 = 왼쪽 어깨
                cv2.circle(img, (lmList[11][1], lmList[11][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 12 = 오른쪽 어깨
                cv2.circle(img, (lmList[12][1], lmList[12][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 13 = 왼쪽 팔꿈치
                cv2.circle(img, (lmList[13][1], lmList[13][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 14 = 오른쪽 팔꿈치
                cv2.circle(img, (lmList[14][1], lmList[14][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 15 = 왼쪽 손목
                cv2.circle(img, (lmList[15][1], lmList[15][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 16 = 오른쪽 손목
                cv2.circle(img, (lmList[16][1], lmList[16][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 23 = 왼쪽 골반
                cv2.circle(img, (lmList[23][1], lmList[23][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 24 = 오른쪽 골반
                cv2.circle(img, (lmList[24][1], lmList[24][2]),
                           15, (0, 0, 250), cv2.FILLED)

                # 오른쪽
                line_R = (math.sqrt(((lmList[14][1] - lmList[12][1]) ** 2) +
                                    ((lmList[14][2] - lmList[12][2]) ** 2)))

                line_R2 = (math.sqrt(((lmList[24][1] - lmList[12][1]) ** 2) +
                                     ((lmList[24][2] - lmList[12][2]) ** 2)))

                line_R3 = (math.sqrt(((lmList[14][1] - lmList[24][1]) ** 2) +
                                     ((lmList[14][2] - lmList[24][2]) ** 2)))

                # 왼쪽
                line_L = (math.sqrt(((lmList[11][1] - lmList[13][1]) ** 2) +
                                    ((lmList[11][2] - lmList[13][2]) ** 2)))

                line_L2 = (math.sqrt(((lmList[11][1] - lmList[23][1]) ** 2) +
                                     ((lmList[11][2] - lmList[23][2]) ** 2)))

                line_L3 = (math.sqrt(((lmList[13][1] - lmList[23][1]) ** 2) +
                                     ((lmList[13][2] - lmList[23][2]) ** 2)))

                sR = (line_R + line_R2 + line_R3) / 2
                sL = (line_L + line_L2 + line_L3) / 2

                SR = (math.sqrt(sR * (sR - line_R) * (sR - line_R2) * (sR - line_R3)))
                SL = (math.sqrt(sL * (sL - line_L) * (sL - line_L2) * (sL - line_L3)))

                Max_R = max(line_R, line_R2, line_R3)
                Max_L = max(line_L, line_L2, line_L3)

                if line_R == Max_R:
                    hR = (2 * SR) / line_R
                    O_R2 = math.asin(hR / line_R3)
                    O = (360) / (np.pi * 2)
                    OR2 = O_R2 * O
                    R = OR2

                elif Max_R == line_R3:
                    hR = (2 * SR) / line_R3
                    O_R = math.asin(hR / line_R)
                    O_R2 = math.asin(hR / line_R2)
                    O = (360) / (np.pi * 2)
                    OR = O_R * O
                    OR2 = O_R2 * O
                    R = 180 - (OR + OR2)

                else:
                    hR = (2 * SR) / line_R2
                    O_R = math.asin(hR / line_R)
                    O = (360) / (np.pi * 2)
                    OR = O_R * O
                    R = OR

                if line_L == Max_L:
                    hL = (2 * SL) / line_L
                    O_L2 = math.asin(hL / line_L3)
                    OL2 = O_L2 * O
                    L = OL2

                elif Max_L == line_L3:
                    hL = (2 * SL) / line_L3
                    O_L = math.asin(hL / line_L)
                    O_L2 = math.asin(hL / line_L2)
                    OL = O_L * O
                    OL2 = O_L2 * O
                    L = 180 - (OL + OL2)

                else:
                    hL = (2 * SL) / line_L2
                    O_L = math.asin(hL / line_L)
                    OL = O_L * O
                    L = OL

                cv2.line(img, (lmList[14][1], lmList[14][2]), (lmList[24][1], lmList[24][2]), (255, 0, 150), 2)

                cv2.line(img, (lmList[13][1], lmList[13][2]), (lmList[23][1], lmList[23][2]), (255, 0, 150), 2)

                cv2.putText(img, str(line_R3), (lmList[14][1] + 40, lmList[14][2] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(img, str(line_L3), (lmList[13][1] + 40, lmList[13][2] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 0), 1, cv2.LINE_AA)

                for i in range(6):
                    l.append(lmList[i + 11][2])
                    j.append(lmList[i + 11][1])
                    cv2.putText(img, ('(' + str(lmList[i + 11][1]) + '), (' + str(lmList[i + 11][2]) + ')'),
                                (lmList[i + 11][1], lmList[i + 11][2] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)

                Max = max(l)
                Min = min(l)
                result = Max - Min
                result_2 = Min - Max

                # 하트 찾는 labels
                if (lmList[11][1] < lmList[13][1] and lmList[13][1] > lmList[15][1]) and (
                        lmList[12][1] > lmList[14][1] and lmList[14][1] < lmList[16][1]) \
                        and (lmList[11][2] > lmList[13][2] > lmList[15][2]) and (
                        lmList[12][2] > lmList[14][2] > lmList[16][2]):
                    cv2.putText(img, 'This pose is a heart', (lmList[4][1] - 60, lmList[4][2] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
                    self.changeText.emit('This pose is a heart')

                    if heart:
                        th = Thread_Heart(self)
                        th.start()
                        print('하트입니다.')
                    heart = False

                else:
                    heart = True

                # 왼쪽 찾는 labelsA
                if (5 < R < 72 and 72 < L < 140) and (lmList[11][1] < lmList[13][1] < lmList[15][1]) \
                        and (lmList[12][2] < lmList[14][2] < lmList[16][2]):
                    cv2.putText(img, 'This pose is the left arm', (lmList[4][1] - 60, lmList[4][2] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
                    self.changeText.emit('This pose is the left arm')

                    if left:
                        th = Thread_Left(self)
                        th.start()
                        print("왼쪽팔 입니다.~")
                    left = False

                else:
                    left = True

                # 오른쪽 찾는 labels
                if (80 < R < 130 and 5 < L < 80) and (lmList[12][1] > lmList[14][1] > lmList[16][1]) \
                        and ((lmList[11][2] < lmList[13][2] < lmList[15][2]) and (lmList[13][2] > lmList[12][2])) \
                        and ((lmList[13][2] > lmList[12][2]) and (lmList[13][2] > lmList[14][2]) and (
                        lmList[13][2] > lmList[16][2])) \
                        and ((lmList[15][2] > lmList[12][2]) and (lmList[15][2] > lmList[14][2]) and (
                        lmList[15][2] > lmList[16][2])):

                    cv2.putText(img, 'This pose is the right arm', (lmList[4][1] - 60, lmList[4][2] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
                    self.changeText.emit('This pose is the right arm')

                    if right:
                        print("오른팔 입니다.~")
                        th = Thread_Right(self)
                        th.start()
                    right = False

                else:
                    right = True

                # 양팔 찾는 labels arms
                if (65 < R < 130 and 65 < L < 130) and \
                        (lmList[16][1] < lmList[14][1] < lmList[12][1] < lmList[11][1] < lmList[13][1] < lmList[15][
                            1]) and \
                        result_2 <= lmList[16][2] - lmList[15][2] <= result and result_2 <= lmList[16][2] - lmList[14][
                    2] <= result \
                        and result_2 <= lmList[16][2] - lmList[13][2] <= result and result_2 <= lmList[16][2] - \
                        lmList[12][2] <= result and result_2 <= lmList[16][2] - lmList[11][2] <= result \
                        and result_2 <= lmList[15][2] - lmList[14][2] <= result and result_2 <= lmList[15][2] - \
                        lmList[13][2] <= result \
                        and result_2 <= lmList[15][2] - lmList[12][2] <= result and result_2 <= lmList[15][2] - \
                        lmList[11][2] <= result \
                        and result_2 <= lmList[14][2] - lmList[13][2] <= result and result_2 <= lmList[14][2] - \
                        lmList[12][2] <= result \
                        and result_2 <= lmList[14][2] - lmList[11][2] \
                        and result_2 <= lmList[13][2] - lmList[12][2] <= result and result_2 <= lmList[13][2] - \
                        lmList[11][2] <= result \
                        and result_2 <= lmList[12][2] - lmList[11][2] <= result:

                    cv2.putText(img, 'This pose is the arms', (lmList[4][1] - 60, lmList[4][2] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)

                    self.changeText.emit('This pose is the arms')

                    if arms:
                        print("양팔 입니다.~")
                        th = Thread_AMSR(self)
                        th.start()
                    arms = False

                else:
                    arms = True

                cTime = time.time()
                fps_rate = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, str(int(fps_rate)), (70, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1920, 1080, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class Thread4(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        global running4
        cap = cv2.VideoCapture(0)
        while running4:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1669, 1233, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class Main_CCTV(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.CCTV_label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle("메인 CCTV")

        menu_group_box = QGroupBox("메뉴")
        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("정지")
        self.stop_button.clicked.connect(self.stop)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save)
        self.end_button = QPushButton("나가기")
        self.end_button.clicked.connect(self.end)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.start_button)
        menu_layout.addWidget(self.stop_button)
        menu_layout.addWidget(self.save_button)
        menu_layout.addWidget(self.end_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        CCTV_group_box = QGroupBox("CCTV")

        self.CCTV_label = QLabel(self)
        self.CCTV_label.move(1669, 1233)
        self.CCTV_label.resize(1669, 1233)

        CCTVLayout = QVBoxLayout()
        CCTVLayout.addWidget(self.CCTV_label)
        CCTVLayout.addStretch(1)
        CCTV_group_box.setLayout(CCTVLayout)

        predictGroupBox = QGroupBox("분류 예측")
        cnt = 0
        self.predict_label = QLabel(self)

        predictLayout = QVBoxLayout()
        predictLayout.addWidget(self.predict_label)
        predictLayout.addStretch(1)
        predictGroupBox.setLayout(predictLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 3)
        mainLayout.addWidget(CCTV_group_box, 1, 0, 2, 2)
        mainLayout.addWidget(predictGroupBox, 1, 2, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def start(self):
        global running
        running = True
        th = Thread1(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.predict_label.setText(f'어린이 : \n어른: \n자전거 : \n킥보드 : ')
        self.show()

    def stop(self):
        global running
        running = False

    def save(self):
        self.predict_label.setText(f'저장하였습니다')

    def end(self):
        exit(0)


class Main_CCTV2(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.CCTV_label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle("CCTV2")
        menu_group_box = QGroupBox("메뉴")
        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("정지")
        self.stop_button.clicked.connect(self.stop)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save)
        self.end_button = QPushButton("나가기")
        self.end_button.clicked.connect(self.end)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.start_button)
        menu_layout.addWidget(self.stop_button)
        menu_layout.addWidget(self.save_button)
        menu_layout.addWidget(self.end_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        CCTV_group_box = QGroupBox("CCTV")

        self.CCTV_label = QLabel(self)
        self.CCTV_label.move(1669, 1233)
        self.CCTV_label.resize(1669, 1233)

        CCTVLayout = QVBoxLayout()
        CCTVLayout.addWidget(self.CCTV_label)
        CCTVLayout.addStretch(1)
        CCTV_group_box.setLayout(CCTVLayout)

        predictGroupBox = QGroupBox("분류 예측")

        self.predict_label = QLabel(self)

        predictLayout = QVBoxLayout()
        predictLayout.addWidget(self.predict_label)
        predictLayout.addStretch(1)
        predictGroupBox.setLayout(predictLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 3)
        mainLayout.addWidget(CCTV_group_box, 1, 0, 2, 2)
        mainLayout.addWidget(predictGroupBox, 1, 2, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def start(self):
        global running2
        running2 = True
        th = Thread2(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

    def stop(self):
        global running2
        running2 = False

    def save(self):
        self.predict_label.setText(f'저장하였습니다')

    def end(self):
        exit(0)


class Pose_Cam(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.x, self.y = 1200, 900

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.CCTV_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def setText(self, text):
        self.predict_label.setText(text)

    # @pyqtSlot(QWidget)
    # def setText(self, change_R, change_L, change_H, change_A):
    #     self.predict_label.setText(QLabel.setText(change_H))
    #     self.predict_label2.setText(QLabel.setText(change_L))
    #     self.predict_label3.setText(QLabel.setText(change_R))
    #     self.predict_label4.setText(QLabel.setText(change_A))

    def initUI(self):
        self.setWindowTitle("Pose_1")
        menu_group_box = QGroupBox("메뉴")

        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start)

        self.stop_button = QPushButton("정지")
        self.stop_button.clicked.connect(self.stop)

        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save)

        self.end_button = QPushButton("나가기")
        self.end_button.clicked.connect(self.end)

        self.size_button = QPushButton("1920X1080")
        self.size_button.clicked.connect(self.size1)

        self.size_button2 = QPushButton("2560X1440")
        self.size_button2.clicked.connect(self.size2)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.start_button)
        menu_layout.addWidget(self.stop_button)
        menu_layout.addWidget(self.save_button)
        menu_layout.addWidget(self.end_button)
        menu_layout.addWidget(self.size_button)
        menu_layout.addWidget(self.size_button2)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        CCTV_group_box = QGroupBox("CAM")

        self.CCTV_label = QLabel(self)

        CCTVLayout = QVBoxLayout()
        CCTVLayout.addWidget(self.CCTV_label)
        CCTVLayout.addStretch(1)
        CCTV_group_box.setLayout(CCTVLayout)

        predictGroupBox = QGroupBox("포즈 예측")

        self.predict_label = QLabel(self)
        self.predict_label2 = QLabel(self)
        self.predict_label3 = QLabel(self)
        self.predict_label4 = QLabel(self)

        predictLayout = QVBoxLayout()
        predictLayout.addWidget(self.predict_label)
        predictLayout.addWidget(self.predict_label2)
        predictLayout.addWidget(self.predict_label3)
        predictLayout.addWidget(self.predict_label4)
        predictLayout.addStretch(1)
        predictGroupBox.setLayout(predictLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 3)
        mainLayout.addWidget(CCTV_group_box, 1, 0, 2, 2)
        mainLayout.addWidget(predictGroupBox, 1, 2, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def start(self):
        global running3
        running3 = True
        th = Thread3(self)
        th.changePixmap.connect(self.setImage)
        th.changeText.connect(self.setText)
        th.start()
        self.show()

    def stop(self):
        global running3
        running3 = False

    def save(self):
        self.predict_label.setText(f'저장하였습니다')

    def end(self):
        exit(0)

    def size1(self):
        self.x = 1200
        self.y = 920
        self.CCTV_label.move(self.x, self.y)
        self.CCTV_label.resize(self.x, self.y)

    def size2(self):

        self.x = 1669
        self.y = 1233
        self.CCTV_label.move(self.x, self.y)
        self.CCTV_label.resize(self.x, self.y)


class Pose_Cam2(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.CCTV_label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle("Pose_CAM2")
        menu_group_box = QGroupBox("메뉴")
        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("정지")
        self.stop_button.clicked.connect(self.stop)
        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self.save)
        self.end_button = QPushButton("나가기")
        self.end_button.clicked.connect(self.end)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.start_button)
        menu_layout.addWidget(self.stop_button)
        menu_layout.addWidget(self.save_button)
        menu_layout.addWidget(self.end_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        CCTV_group_box = QGroupBox("CAM")

        self.CCTV_label = QLabel(self)
        self.CCTV_label.move(1669, 1233)
        self.CCTV_label.resize(1669, 1233)

        CCTVLayout = QVBoxLayout()
        CCTVLayout.addWidget(self.CCTV_label)
        CCTVLayout.addStretch(1)
        CCTV_group_box.setLayout(CCTVLayout)

        predictGroupBox = QGroupBox("포즈 예측")

        self.predict_label = QLabel(self)

        predictLayout = QVBoxLayout()
        predictLayout.addWidget(self.predict_label)
        predictLayout.addStretch(1)
        predictGroupBox.setLayout(predictLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 3)
        mainLayout.addWidget(CCTV_group_box, 1, 0, 2, 2)
        mainLayout.addWidget(predictGroupBox, 1, 2, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def start(self):
        global running4
        running4 = True
        th = Thread4(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.predict_label.setText(f'어린이 : \n어른: \n자전거 : \n킥보드 : ')
        self.show()

    def stop(self):
        global running4
        running4 = False

    def save(self):
        self.predict_label.setText(f'저장하였습니다')

    def end(self):
        exit(0)


class ClassificationAITab(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('School_Zone')

        CCTV = Main_CCTV()
        CCTV2 = Main_CCTV2()
        Cam = Pose_Cam()
        Cam2 = Pose_Cam2()

        tabs = QTabWidget()
        tabs.addTab(CCTV, '메인CCTV')
        tabs.addTab(CCTV2, '서브CCTV2')
        tabs.addTab(Cam, '포즈Cam')
        tabs.addTab(Cam2, '포즈Cam2')

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)

        self.setLayout(vbox)


if __name__ == '__main__':
    running, running2, running3, running4, heart, arms, left, right =\
        False, False, False, False, False, False, False, False
    A = 'This pose is the arms'
    H = 'This pose is a heart'
    L = 'This pose is the left arm'
    R = 'This pose is the right arm'
    app = QApplication(sys.argv)
    ex = ClassificationAITab()
    ex.show()
    sys.exit(app.exec_())
