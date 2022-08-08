import time, torch, winsound, os, math, threading
from pathlib import Path
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_imshow, cv2, non_max_suppression, scale_coords,)
from utils.plots import Annotator, colors
from playsound import playsound

if not os.path.exists('image_data/'):
    os.mkdir('image_data/')


# 자동차
def execute():
    threading.currentThread().getName(), playsound('mp3file/fcar.mp3')
    pass


# 자전거
def execute2():
    threading.currentThread().getName(), playsound('mp3file/bick.mp3')


# 킥보드
def execute3():
    threading.currentThread().getName(), playsound('mp3file/kick.mp3')


# 킥보드_자전거
def execute4():
    threading.currentThread().getName(), playsound('mp3file/kick-bick.mp3')


car = True
bick = True
kick = True
kick_bick = True

## CCTV를 이용한 객체인식 ##
# source = 'image_data/test.mp4' # 저화질
# source = 'image_data/test2.mp4' # 고화질
# source = 'image_data/kim.mp4' # 유튜브 영상
source = 'image_data/2.mp4' #1차 교내 대회 영상
# source = 'image_data/22.mp4' # 두현쌤 이 주신 신호등
# source = 'image_data/unity02.mov'

# source = 'image_data/13.png'

## CCTV를 이용한 객체인식 ##

## 카메라를 이용한 포즈 인식 ##

## 카메라를 이용한 포즈 인식 ##

imgsz = (640, 640)
device = torch.device('cpu')

# 학습된 모델 불러오기
weights = ['Name_Models/Object/Object.pt']
weights2 = ['Name_Models/Kid_Adult/Kid_Adult_best (1).pt']
weights3 = ['Name_Models/light/light_best.pt']

# 데이터 야물 파일 불러오기
data = 'Name_Yaml/Object/Object.yaml'
data2 = 'Name_Yaml/Kid_Adult/Kid_Adult_data2.yaml'
data3 = 'Name_Yaml/light/light_data.yaml'

dnn = False
fp16 = False
fp32 = False
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
line_thickness = 3
hide_labels = False  # hide labels
hide_conf = False

model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=fp16)
model2 = DetectMultiBackend(weights2, device=device, dnn=dnn, data=data2, fp16=fp16)
model3 = DetectMultiBackend(weights3, device=device, dnn=dnn, data=data3, fp16=fp16)

stride, names, pt = model3.stride, model.names, model3.pt
stride2, names2, pt2 = model2.stride, model2.names, model2.pt
stride3, names3, pt3 = model3.stride, model3.names, model3.pt

names.append('Kid')
names.append('Adult')
names.append('Cross')
names.append('Cross')
names.append('R_Signal')
names.append('G_Signal')

imgsz = check_img_size(imgsz, s=stride)  # check image size
view_img = check_imshow()

dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
bs = 1

model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
model2.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # //
model3.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # //

dt, seen = [0.0, 0.0, 0.0], 0

for (path, im, im0s, vid_cap, s) in dataset:
    t1 = time.time()
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 and model2.fp16 and model3.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time.time()
    dt[0] += t2 - t1

    pred = model(im)
    pred2 = model2(im)
    pred3 = model3(im)

    t3 = time.time()
    dt[1] += t3 - t2

    # 불필요한 네모박스 삭제
    pr = non_max_suppression(pred, conf_thres, iou_thres, classes=None, max_det=max_det)
    pr2 = non_max_suppression(pred2, conf_thres, iou_thres, classes=None, max_det=max_det)
    pr3 = non_max_suppression(pred3, conf_thres, iou_thres, classes=None, max_det=max_det)

    dt[2] += time.time() - t3

    for (i, det), (i_2, det_2), (i_3, det_3) in zip(enumerate(pr), enumerate(pr2), enumerate(pr3)):  # per image
        seen += 1
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        imc = im0.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):
                if view_img:
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
        # 어린이 성인
        if len(det_2):
            # Rescale boxes from img_size to im0 size
            det_2[:, :4] = scale_coords(im.shape[2:], det_2[:, :4], im0.shape).round()
            Kid_Adult = []

            Kid_Adult.append('Kid')
            Kid_Adult.append('Adult')
            Kid_Adult.append('Cross')

            # Print results
            for c in det_2[:, -1].unique():
                n = (det_2[:, -1] == c).sum()  # detections per class
                s += f"{n} {Kid_Adult[0]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det_2):
                if view_img:
                    c = int(cls)  # integer class
                    label = None if hide_labels else (Kid_Adult[c] if hide_conf else f'{Kid_Adult[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
        # 횡단보도, 빨간불, 파란불
        if len(det_3):
            # Rescale boxes from img_size to im0 size
            det_3[:, :4] = scale_coords(im.shape[2:], det_3[:, :4], im0.shape).round()
            light = []

            light.append('Cross')
            light.append('R_Signal')
            light.append('G_Signal')

            # Print results
            for c in det_3[:, -1].unique():
                n = (det_3[:, -1] == c).sum()  # detections per class
                s += f"{n} {light[0]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det_3):
                if view_img:
                    c = int(cls)  # integer class
                    label = None if hide_labels else (light[c] if hide_conf else f'{light[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        min_list = []
        e = []
        if view_img:

            for o in range(len(annotator.kick)):
                if annotator.kick[o] == 1:
                    if kick:
                        my_thread = threading.Thread(target=execute3())
                        my_thread.start()
                    kick = False
                else:
                    kick = True

            for o in range(len(annotator.bick)):
                if annotator.bick[o] == 1:
                    if kick:
                        my_thread = threading.Thread(target=execute2())
                        my_thread.start()
                    bick = False
                else:
                    bick = True

            for o in range(len(annotator.kick_bick)):
                if annotator.kick_bick[o] == 1:
                    if kick:
                        my_thread = threading.Thread(target=execute3())
                        my_thread.start()
                    kick_bick = False
                else:
                    kick_bick = False

            # 사람 vs 자동차
            if True:
                a = []
                b = []
                c = []
                d = []
                thickness = int(2)
                for k in range(len(annotator.Kid_X)):
                    # cv2.rectangle(im0, annotator.person_box[k * 2], annotator.person_box[k * 2 + 1], (255, 0, 0), 5)
                    # person
                    for j in range(len(annotator.car)):
                        # 직선의 거리를 구하는 코드
                        a.append(int(math.sqrt(((annotator.Kid_X[k] - annotator.car_X[j]) ** 2) +
                                               ((annotator.Kid_Y[k] - annotator.car_Y[j]) ** 2))))
                        min_value = min(a)
                        min_list.append(min_value)
                        if min_value < 160:
                            if car:
                                my_thread = threading.Thread(target=execute())
                                my_thread.start()
                            car = False
                        else:
                            car = True
                            # print("자동차 근처에 사람이 있어요.", min_value)

                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.Kid[k], annotator.car[j], (0, 0, 255), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(a[k]),
                                          (int((int(annotator.car_X[j]) + int(annotator.Kid_X[k])) / 2),
                                           int((int(annotator.car_Y[j]) + int(annotator.Kid_Y[k])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)
                    # 버스
                    for j in range(len(annotator.bus)):
                        # 직선 길이 구하는 코드
                        b.append(int(math.sqrt(((annotator.Kid_X[k] - annotator.bus_X[j]) ** 2) +
                                               ((annotator.Kid_Y[k] - annotator.bus_Y[j]) ** 2))))
                        min_value = min(b)
                        min_list.append(min_value)
                        if min_value < 120:
                            kkkk = 0
                            # print("버스 근처에 사람이 있어요.", min_value)
                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.Kid[k], annotator.bus[j], (0, 255, 255), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(b[k]),
                                          (int((int(annotator.bus_X[j]) + int(annotator.Kid_X[k])) / 2),
                                           int((int(annotator.bus_Y[j]) + int(annotator.Kid_Y[k])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)
                    # truck
                    for j in range(len(annotator.truck)):
                        # 직선의 거리를 구하는 코드
                        c.append(int(math.sqrt(((annotator.Kid_X[k] - annotator.truck_X[j]) ** 2) +
                                               ((annotator.Kid_Y[k] - annotator.truck_Y[j]) ** 2))))
                        min_value = min(c)
                        min_list.append(min_value)
                        if min_value < 120:
                            kkkk = 0
                            # print("트럭 근처에 사람이 있어요.", min_value)
                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.Kid[k], annotator.truck[j], (0, 255, 0), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(c[k]),
                                          (int((int(annotator.truck_X[j]) + int(annotator.Kid_X[k])) / 2),
                                           int((int(annotator.truck_Y[j]) + int(annotator.Kid_Y[k])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)

                for h in range(len(annotator.person_X)):
                    # cv2.rectangle(im0, annotator.person_box[h*2], annotator.person_box[h*2+1], (255, 0, 0), 5)
                    # person
                    for j in range(len(annotator.car)):
                        # 직선의 거리를 구하는 코드
                        a.append(int(math.sqrt(((annotator.person_X[h] - annotator.car_X[j]) ** 2) +
                                               ((annotator.person_Y[h] - annotator.car_Y[j]) ** 2))))
                        min_value = min(a)
                        min_list.append(min_value)
                        if min_value < 160:
                            if car:
                                my_thread = threading.Thread(target=execute())
                                my_thread.start()
                                print("자동차 근처에 사람이 있어요.", min_value)
                            car = False
                        else:
                            car = True

                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.person[h], annotator.car[j], (0, 0, 255), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(a[h]),
                                          (int((int(annotator.car_X[j]) + int(annotator.person_X[h])) / 2),
                                           int((int(annotator.car_Y[j]) + int(annotator.person_Y[h])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)
                    # 버스
                    for j in range(len(annotator.bus)):
                        # 직선 길이 구하는 코드
                        b.append(int(math.sqrt(((annotator.person_X[h] - annotator.bus_X[j]) ** 2) +
                                               ((annotator.person_Y[h] - annotator.bus_Y[j]) ** 2))))
                        min_value = min(b)
                        min_list.append(min_value)
                        if min_value < 120:
                            kkkk = 0
                            # print("버스 근처에 사람이 있어요.", min_value)
                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.person[h], annotator.bus[j], (0, 255, 255), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(b[h]),
                                          (int((int(annotator.bus_X[j]) + int(annotator.person_X[h])) / 2),
                                           int((int(annotator.bus_Y[j]) + int(annotator.person_Y[h])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)
                    # truck
                    for j in range(len(annotator.truck)):
                        # 직선의 거리를 구하는 코드
                        c.append(int(math.sqrt(((annotator.person_X[h] - annotator.truck_X[j]) ** 2) +
                                               ((annotator.person_Y[h] - annotator.truck_Y[j]) ** 2))))
                        min_value = min(c)
                        min_list.append(min_value)
                        if min_value < 120:
                            kkkk = 0
                            # print("트럭 근처에 사람이 있어요.", min_value)
                        # 직선 그리는 코드
                        im0 = cv2.line(im0, annotator.person[h], annotator.truck[j], (0, 255, 0), 1)
                        # 텍스트 그리는 코드
                        im0 = cv2.putText(im0, str(c[h]),
                                          (int((int(annotator.truck_X[j]) + int(annotator.person_X[h])) / 2),
                                           int((int(annotator.truck_Y[j]) + int(annotator.person_Y[h])) / 2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (255, 0, 0), thickness, cv2.LINE_AA)
            # im0 = cv2.resize(im0, (1600, 800))


            cv2.imshow("dd", im0)
            # cv2.imwrite('image_data/22.jpg', im0)
            # for ppp in range(20):
            #     cv2.imwrite('image_data/image_data/' + str(ppp) + '.jpg', im0)
    if cv2.waitKey(1) == ord('q'):
        break
