# 기본적으로 이미지 처리를 위해 cv2, 저장하기 위해 sys, 이미지 확인을 위해 matplolib, 리스트 배열을 처리해주기 위해 numpy
import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/300.jpg")
image_gray = cv2.imread("images/300.jpg", cv2.IMREAD_GRAYSCALE)

# plt.imshow(image)
# plt.show()
# plt.imshow(image_gray)
# plt.show()

b,g,r = cv2.split(image)
image2 = cv2.merge([r,g,b])
 
# plt.imshow(image2)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# 블러는 이미지를 부드럽게 하기 때문에 배경과 원하지 않는 부분을 부드럽게 하여 해당 경계를 찾지 못하게 하기 위함이다. 
# 블러 처리를 심하게 하면 원래 찾고자 하는 대상을 찾지 못하게 되고, 블러가 약하면 배경의 외곽도 포함되므로 ksize를 적당히 조절 해야 한다. 
blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
plt.imshow(blur)
plt.show()

'''
blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
plt.imshow(blur)
plt.show()
'''

edged = cv2.Canny(blur, 10, 250)
# cv2.imshow('Edged', edged)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed', closed)
# cv2.waitKey(0)


# contours 물체가 몇개인지 
contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
print(contours)

# 외곽선 그리는 용도. 이미지에 그리기 때문에 이 코드 적용하면 원래 이미지에
# 초록색 선 생김
contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('contours_image', contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





# https://youbidan.tistory.com/19