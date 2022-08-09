
<p align=center>
<img src="https://user-images.githubusercontent.com/26598708/63312262-e856cc00-c33b-11e9-8d1a-61c1098c5e61.png" width="70%"></img>
</p>

## 개발 의도
아이들을 지키기 위한 스쿨존에서 사고가 제일 많이 발생하여, 스쿨존 교통사고 예방은 어린이안전교육 확대가 우선이라고 생각하여 개발함.

## 문제점 분석
어린이안전교육이 부족하며, 현장에서 배우는 것이 아닌 이론으로 배우기 때문에 현장에서 사고가 많으며 돌발행동이 많음

## 개발
PyCharm프로그램으로 포즈인식 및 객체인식을 진행함.

핵심 구현을 위해 추가로 작성한 C# 코드 파일들

`OverlayRing.cs`
`SettingManager.cs`
`UI_Manager.cs`

OverlayRing.cs는 HOTK_TrackedDevice.cs로 회전하는 carmera 오브젝트에 따라 값을 받는다.

##  결과
<img src ="https://github.com/tnsgud9/VR-Overlay-Half_Ring/blob/master/Assets/Half-Ring/Sprites/gif/1.gif?raw=true" width="30%"></img>
<img src ="https://github.com/tnsgud9/VR-Overlay-Half_Ring/blob/master/Assets/Half-Ring/Sprites/gif/1.gif?raw=true" width="30%"></img>
<img src ="https://github.com/tnsgud9/VR-Overlay-Half_Ring/blob/master/Assets/Half-Ring/Sprites/gif/1.gif?raw=true" width="30%"></img>
<img src ="https://github.com/tnsgud9/VR-Overlay-Half_Ring/blob/master/Assets/Half-Ring/Sprites/gif/1.gif?raw=true" width="30%"></img>

### 핵심코드
##### OverlayRing.cs
<pre><code>
    //Important : It is handled first. (Maximum Priority in define to Script Execution Order)
    //중요 : 가장 먼저 처리된다.( 최고 우선 순위로 Script Execution Order에 지정되어 있다. )
    void Update () {
        //Debug.Log("HMD Rotation : " + carmera.transform.eulerAngles);
        overlayOBJ.gameObject.transform.rotation = Quaternion.Euler(0 , 0, 360f-carmera.transform.eulerAngles.z);
    }</code></pre>

## 추가 예정
계절 및 날씨에 따른 결과들을 분석으로 상황에 맞는 인공지능을 만들 계획
예) 비가오는 날이면, 우산을 인식함.

</p>
