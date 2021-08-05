import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/300.jpg")

# plt.figure(figsize=(15,12))
# plt.imshow(img)
# plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.figure(figsize=(15,12))
# plt.imshow(img_gray)
# plt.show()

# 블러는 이미지를 부드럽게 하기 때문에 배경과 원하지 않는 부분을 부드럽게 하여 해당 경계를 찾지 못하게 하기 위함이다. 
# 블러 처리를 심하게 하면 원래 찾고자 하는 대상을 찾지 못하게 되고, 블러가 약하면 배경의 외곽도 포함되므로 ksize를 적당히 조절 해야 한다. 
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) # (이미지, ksize, sigma)
# plt.figure(figsize=(15,12))
# plt.imshow(img_blur)
# plt.show()

# https://m.blog.naver.com/samsjang/220504782549
# cv.threshold(img, threshold_value, value, flag)
# img : Grayscale 이미지
# threshold_value : 픽셀 문턱값
# value : 픽셀 문턱값 보다 클 때 적용되는 최대값 (적용되는 플래그에 따라 픽셀 문턱값보다 작을 때 적용되는 최대값)
# flag : 문턱값 적용 방법 또는 스타일 (THRESH_BINARY, THRESH_BINARY_INV. THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
ret, img_th = cv2.threshold(img_blur, 100, 230, cv2.THRESH_BINARY_INV)  # cv2.THRESH_BINARY_INV => 픽셀값이 threshold_value 보다 크면 value, 작으면 value로 할당

contours, hierachy= cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(each) for each in contours]
# print(rects)

tmp = [w*h for (x,y,w,h) in rects]
tmp.sort()
# print(tmp)

rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>15000)and(w*h<500000))]
for i in rects:
    print(i)

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), 
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 

plt.figure(figsize=(15,12))
plt.imshow(img)
plt.show()




