import sys
import numpy as np
import cv2

# 이미지 읽어오기 
img = cv2.imread("images/test.jpg")

# 그레이스케일로 변환하고 블러걸고 이진화하기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# 윤곽 추출 -> 인덱스를 [0]으로 해줘야함 또는 두개의 변수 기입
# contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 추출한 윤곽 반복처리
for each in contours:
    x, y, w, h = cv2.boundingRect(each)
    if h < 20 : continue # 너무 작으면 건너뛰기 
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), h, 2)
    
cv2.imwrite("images/number-ocr.png", img)
