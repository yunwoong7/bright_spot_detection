# Bright Spot Detection
이미지 연산 방법을 통해 영역을 찾는 방법 중 세번째로 **객체의 빛을 이용하여 검출하는 방법**에 대해 알아 보겠습니다. 앞서 소개한 이미지 연산을 통한 Object Detection 방법(Shape, Color)을 포함하여 이 글에서 다루는 Bright spot detection 까지 매우 단순한 방법입니다. 이 자체만으로는 현재 직면하고 계신 문제를 풀 수 없을지도 모릅니다. 하지만 이런 기능들로부터 영감을 받아 고민하고 응용한다면 꽤 훌륭한 결과물을 만들 수도 있을 것이라 생각합니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/cn36Si/btrR3CTzKCP/d6y1kGkpZbDlh2OurUd1NK/img.gif" width="50%">
</div>

------

#### **Import packages**

```python
import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

이미지에서 빛 영역의 contours를 찾는 Function 정의

```python
def bright_spot(img, w_pixel_cnt=100):
    # 배경색은 검정(0)으로 하고 연결된 구성요소에 label을 붙임
    labels = measure.label(img, connectivity=2, background=0)
    mask = np.zeros(img.shape, dtype="uint8")
 
    for label in np.unique(labels):
        # 배경 label이면 무시
        if label == 0:
            continue
            
        # 배경이 아니면 label mask를 구성
        label_mask = np.zeros(img.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        # 구성요소의 픽셀 수가 임계값보다 큰 경우 mask에 추가
        if num_pixels > w_pixel_cnt:
            mask = cv2.add(mask, label_mask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    
    return cnts
```

#### **Load Image**

```python
cv2_image = cv2.imread('asset/images/star_light.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

테스트 할 이미지는 망원경으로 관찰 한 별 이미지입니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bzfnDF/btrRZLEGzzt/kEARdwEeRQVKdtDJ0O6pHk/img.png" width="50%">
</div>

#### **Bright Spot Detection**

먼저 간단하게 cv2.minMaxLoc 를 이용하여 이미지에서 가장 밝은 픽셀을 찾아 빛의 영역을 찾을 수도 있습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bpZ16A/btrR2HuDPgk/fpZNMIeRFekJLKSkTUrXgK/img.png" width="50%">
</div>

```python
spot_img = cv2_image.copy()
gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (13, 13), 0)
 
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
spot_img = cv2.circle(spot_img, maxLoc, 90, (0, 255, 0), 5)
 
img_show('bright spot image', spot_img)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/GGKN7/btrR3bviNIY/ijv95KKI2cMWFpoFzEWpzk/img.png" width="50%">
</div>

cv2.minMaxLoc만 이용하더라도 매우 쉽게 찾을 수도 있습니다. 하지만 cv2.minMaxLoc는 노이즈에 매우 취약하기때문에 충분한 전처리가 되지 않으면 이 기능만으로는 부족합니다. 그리고 찾은 것은 밝은 영역이 아닌 픽셀이라는 것을 잊어서는 안됩니다. (그렇지만 이미지에서 밝은 단일 픽셀을 찾는 문제라면 너무 간단히 해결하겠죠.)

아래 과정을 통해 이미지를 그레이스케일로 변환하고 노이즈를 줄이기 위한 이미지 블러링 후 이진화 합니다. 노이즈가 있기때문에 침식과 팽창 연산을 통해 전처리를 진행하였습니다.

```python
gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]
erode = cv2.erode(thresh, None, iterations=2)
dilate = cv2.dilate(erode, None, iterations=2)
 
img_show(['blurred', 'thresh', 'erode', 'dilate'], [blurred, thresh, erode, dilate], figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/9GRRZ/btrR1oWH8vX/Rl1kdMIJh70Rttb7xtIriK/img.png" width="50%">
</div>

bright_spot Function을 통해 contours를 추출하고 원 영역으로 표시합니다.

```python
vis = cv2_image.copy()
 
cnts = bright_spot(dilate, 10)
 
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(vis, (int(cX), int(cY)), int(radius), (0, 255, 0), 3)
```

Bright spot을 표현한 이미지를 확인합니다.

```python
img_show('Bright spot image', vis)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/MNMEe/btrR33XPp2n/WPDwGi9n2C44dKCuhRlOm0/img.png" width="50%">
</div>

------

위에서 언급했듯이 이 기능은 단순한 이미지 연산을 통한 Object Detection 입니다. 이미지 노이즈에 따라 결과가 매우 달라지고 전처리, 임계값, 설정값 등에 따라 다른 결과를 보입니다. 이 기능 자체로 사용한다기보다는 응용하여 사용하면 새로운 기능을 발견하거나 아이디어를 얻는데 도움이 될 것 같습니다.

예를들면, 제조 과정을 검수 단계에서 이물질을 검수하거나 적외선 이미지에서 빛(산불, 폭발 등)을 찾는 문제에 활용하면 좋을 것 같습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/zHaWu/btrR2ESlkv4/wJ6dfsOE8EQDY3maKLkkQK/img.png" width="50%">
</div>
