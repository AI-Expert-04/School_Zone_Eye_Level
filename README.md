
<p align=center>
<img src="" width="70%"></img>
</p>

## 개발 의도
아이들을 지키기 위한 스쿨존에서 사고가 제일 많이 발생하여, 스쿨존 교통사고 예방은 어린이안전교육 확대가 우선이라고 생각하여 개발함.

## 문제점 분석
어린이안전교육이 부족하며, 현장에서 배우는 것이 아닌 이론으로 배우기 때문에 현장에서 사고가 많으며 돌발행동이 많음

## 개발
PyCharm프로그램으로 포즈인식 및 객체인식을 진행함.

## 파일구조
```
CCTV_Detecter
 |---- yolov5
 |    |---- Name_Models
 |    |    |---- Kid_Adult
 |    |    |    |---- Kid_Adult_best (1).pt  -> 어린이_어른 구별 모델
 |    |    |    |---- Kid_Adult_best.pt
 |    |    |    |---- Kid_Adult_last.pt
 |    |    |---- Object
 |    |    |    |---- Object.pt -> 자동차, 버스, 트럭, 사람 감지 모델
 |    |    |---- light
 |    |    |    |---- light_best.pt -> 도로, 빨간불, 파란불 감지 모델
 |    |    |    |---- light_last.pt
 |    |---- Name_Yaml
 |    |    |---- Kid_Adult
 |    |    |    |---- Kid_Adult_data.yaml -> class_name[어린이, 어른]
 |    |    |    |---- Kid_Adult_data2.yaml
 |    |    |---- Object
 |    |    |    |---- Object.yaml -> class_name[자동차, 버스, 트럭, 사람]
 |    |    |---- light
 |    |    |    |---- light_data.yaml -> class_name[도로, 빨간불, 파란불]
 |    |---- mp3file
 |    |    |    |---- bick.mp3--------|
 |    |    |    |---- fcar.mp3--------|________ 상황에 맞는 소리
 |    |    |    |---- kick-bick.mp3---|
 |    |    |    |---- kick.mp3--------|
 |    |---- CCTV_Detecter.py
```

핵심 구현을 위해 추가로 작성한 Pyhon 코드 파일들

`CCTV_Detecter.py`

OverlayRing.cs는 HOTK_TrackedDevice.cs로 회전하는 carmera 오브젝트에 따라 값을 받는다.

##  결과
<img src ="" width="30%"></img>
<img src ="" width="30%"></img>
<img src ="" width="30%"></img>
<img src ="" width="30%"></img>

### 핵심코드
##### CCTV_Detecter.py
<pre><code>
    
    </code></pre>

## 추가 예정
계절 및 날씨에 따른 결과들을 분석으로 상황에 맞는 인공지능을 만들 계획
예) 비가오는 날이면, 우산을 인식함.

</p>
