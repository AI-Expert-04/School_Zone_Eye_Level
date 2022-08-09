
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
PLARD
 |---- ptsemseg
 |---- imgs
 |---- outputs
 |---- dataset
 |    |---- training
 |    |    |---- image_2
 |    |    |---- ADI
 |    |---- testing
 |    |    |---- image_2
 |    |    |---- ADI 
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
