import mediapipe as mp
import numpy as np
import time, os, math, threading, cv2
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

# 쓰레드


class Threads():
    def __init__(self):
        self.threading = threading
        self.playsound = playsound

    # 하트
    def execute(self):
        self.threading.currentThread().getName(), self.playsound('mp3-file/hello_school.mp3')

    # 양팔
    def execute2(self):
        self.threading.currentThread().getName(), self.playsound('mp3-file/blue.mp3')

    # 왼팔
    def execute3(self):
        self.threading.currentThread().getName(), self.playsound('mp3-file/left.mp3')

    # 오른팔
    def execute4(self):
        self.threading.currentThread().getName(), self.playsound('mp3-file/right.mp3')


heart = True
arms = True
left = True
right = True


def main():
    global heart, arms, left, right
    # cap = cv2.VideoCapture('image_2/pose03.mp4')
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('image_2/1 (2).mp4')
    pTime = 0
    detector = poseDetection()
    l = []
    j = []

    while True:
        success, img = cap.read()
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

        AB_R = (lmList[14][1] * lmList[24][1]) + (lmList[14][2] * lmList[24][2])
        AB_L = (lmList[13][1] * lmList[23][1]) + (lmList[13][2] * lmList[23][2])

        print(line_R, line_R2, line_R3)

        sR = (line_R + line_R2 + line_R3) / 2
        sL = (line_L + line_L2 + line_L3) / 2

        print(sL)
        print(line_L, line_L2, line_L3)

        SR = (math.sqrt(sR * (sR - line_R) * (sR - line_R2) * (sR - line_R3)))
        SL = (math.sqrt(sL * (sL - line_L) * (sL - line_L2) * (sL - line_L3)))

        Max_R = max(line_R, line_R2, line_R3)
        Max_L = max(line_L, line_L2, line_L3)

        print(Max_R, Max_L)

        if line_R == Max_R:
            hR = (2 * SR) / line_R
            O_R = math.asin(hR / line_R2)
            O_R2 = math.asin(hR / line_R3)
            O = (360) / (np.pi * 2)
            OR2 = O_R2 * O
            R = OR2
            print(1)

        elif Max_R == line_R3:
            hR = (2 * SR) / line_R3
            O_R = math.asin(hR / line_R)
            O_R2 = math.asin(hR / line_R2)
            O = (360) / (np.pi * 2)
            OR = O_R * O
            OR2 = O_R2 * O
            R = 180 - (OR + OR2)
            print(2)

        else:
            hR = (2 * SR) / line_R2
            O_R = math.asin(hR / line_R)
            O_R2 = math.asin(hR / line_R3)
            O = (360) / (np.pi * 2)
            OR = O_R * O
            R = OR
            print(3)

        if line_L == Max_L:
            hL = (2 * SL) / line_L
            O_L = math.asin(hL / line_L2)
            O_L2 = math.asin(hL / line_L3)
            OL2 = O_L2 * O
            L = OL2
            print('L', 1)

        elif Max_L == line_L3:
            hL = (2 * SL) / line_L3
            O_L = math.asin(hL / line_L)
            O_L2 = math.asin(hL / line_L2)
            OL = O_L * O
            OL2 = O_L2 * O
            print(OL, OL2)
            L = 180 - (OL + OL2)
            print('L', 2)

        else:
            hL = (2 * SL) / line_L2
            O_L = math.asin(hL / line_L)
            O_L2 = math.asin(hL / line_L3)
            OL = O_L * O
            L = OL
            print('L', 3)

        print(R, L)

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
            print(lmList[i + 11])
        Max = max(l)
        Min = min(l)
        result = Max - Min
        result_2 = Min - Max

        # 하트 찾는 labels
        if (lmList[11][1] < lmList[13][1] and lmList[13][1] > lmList[15][1]) and (
                lmList[12][1] > lmList[14][1] and lmList[14][1] < lmList[16][1])\
                and (lmList[11][2] > lmList[13][2] > lmList[15][2]) and (lmList[12][2] > lmList[14][2] > lmList[16][2]):
            cv2.putText(img, 'This pose is a heart', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            if heart:
                my_thread = threading.Thread(target=Threads.execute)
                my_thread.start()

            print('양팔입니다.')
            heart = False

        else:
            cv2.putText(img, 'Find_Pose', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            heart = True

        # 왼쪽 찾는 labelsA
        if (5 < R < 72 and 72 < L < 140) and (lmList[11][1] < lmList[13][1] < lmList[15][1])\
                and (lmList[12][2] < lmList[14][2] < lmList[16][2]):
            cv2.putText(img, 'This pose is the left arm', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            print("왼쪽팔 입니다.~")

            if left:
                my_thread3 = threading.Thread(target=Threads.execute3)
                my_thread3.start()
            left = False

        else:
            cv2.putText(img, 'Find_Pose', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            left = True

        # 오른쪽 찾는 labels
        if (80 < R < 130 and 5 < L < 80) and (lmList[12][1] > lmList[14][1] > lmList[16][1])\
                and ((lmList[11][2] < lmList[13][2] < lmList[15][2]) and (lmList[13][2] > lmList[12][2])) \
                        and ((lmList[13][2] > lmList[12][2]) and (lmList[13][2] > lmList[14][2]) and (
                        lmList[13][2] > lmList[16][2])) \
                        and ((lmList[15][2] > lmList[12][2]) and (lmList[15][2] > lmList[14][2]) and (
                        lmList[15][2] > lmList[16][2])):

            cv2.putText(img, 'This pose is the right arm', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            print("오른팔 입니다.~")

            if right:
                my_thread4 = threading.Thread(target=Threads.execute4)
                my_thread4.start()
            right = False

        else:
            cv2.putText(img, 'Find_Pose', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            right = True

        # 양팔 찾는 labels arms
        if (70 < R < 130 and 70 < L < 130) and \
                (lmList[16][1] < lmList[14][1] < lmList[12][1] < lmList[11][1] < lmList[13][1] < lmList[15][1]) and \
                result_2 <= lmList[16][2] - lmList[15][2] <= result and result_2 <= lmList[16][2] - lmList[14][2] <= result \
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
            print("양팔 입니다.~")

            if arms:
                my_thread2 = threading.Thread(target=Threads.execute2)
                my_thread2.start()
            arms = False

        else:
            cv2.putText(img, 'Find_Pose', (lmList[4][1] - 60, lmList[4][2] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 150), 1, cv2.LINE_AA)
            arms = True


        cTime = time.time()
        fps_rate = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps_rate)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("pose detection", img)

        if cv2.waitKey(40) == ord("q"):
            break


if __name__ == '__main__':

    main()
    cv2.destroyAllWindows()
    # cap.release()