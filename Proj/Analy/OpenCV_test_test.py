# 기본적으로 이미지 처리를 위해 cv2, 저장하기 위해 sys, 이미지 확인을 위해 matplolib, 리스트 배열을 처리해주기 위해 numpy

import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

image = cv2.imread("images/test.jpg")
image_gray = cv2.imread("images/test.jpg", cv2.IMREAD_GRAYSCALE)

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


# contours[] => 물체가 몇 개인지 나타냅니다. 하나의 닫힌 선이 물체 하나를 인식합니다. 위에서는 다른 것을 인식하지 않고 향수병만 인식했으므로 물체는 한 개 입니다. 
# contours[][] => 이 때부터 각 좌표를 나타냅니다
# contours[][][x][y] -> 세번째와 네번째 부터는 각각 x축과 y축을 나타냅니다. 
contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
# print(contours)

# 배경 색 검정으로
img_binary = cv2.bitwise_not(image_gray)

# 외곽선 그리는 용도. 이미지에 그리기 때문에 이 코드 적용하면 원래 이미지에
# 초록색 선 생김
# cv2.drawContours(이미지, [윤곽선], 윤곽선 인덱스, (B, G, R), 두께, 선형 타입)
contours_image = cv2.drawContours(img_binary, contours, -1, (255,255,255), 3)
cv2.imshow('contours_image', contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours_xy = np.array(contours)
contours_xy.shape

model = tf.keras.models.load_model('mnist(CNN).hdf5')

# cv2.boundingRect()함수는 인자로 받은 contour에 외접하고 똑바로 세워진 직사각형의 좌상단 꼭지점 좌표(x,y)와 가로,세로 폭을 리턴
for each in contours:
    x , y, w, h = cv2.boundingRect(each)
    # cv2.rectangle(contours_image, (x,y), (x+w, y+h), (0,0,255), 3)
    
    # cv2.imshow('rectangle', contours_image)
    # cv2.waitKey(0)
    test_cut = contours_image[y:y+h, x:x+w]
    print(test_cut)
    
    test_cut = cv2.resize(255-test_cut, (28, 28))
    test_my_img = test_cut.flatten() / 255.0
    #print(test_my_img)
    # 배경이 0으로 되어있는 모델이라 변경
    # test_my_img = np.where(test_my_img == 1, 2, test_my_img)
    # test_my_img = np.where(test_my_img == 0, 1, test_my_img)
    # test_my_img = np.where(test_my_img == 2, 0, test_my_img)
    #print(test_my_img)
    
    test_my_img = test_my_img.reshape((-1, 28, 28, 1))
    print(test_my_img)
    print('The Answer is ', model.predict_classes(test_my_img))

    cv2.imshow('cut',test_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # https://pythonq.com/so/c%2B%2B/550878
    

# (514, 590, 59, 79)
# (643, 586, 76, 94)
# (385, 582, 107, 98)
# (774, 578, 57, 122)
# (427, 396, 31, 93)
# (661, 380, 86, 93)
# (519, 376, 66, 111)
# (787, 371, 63, 116)
# (417, 161, 30, 129)
# (694, 156, 22, 116)
# (511, 155, 80, 125)
# (767, 146, 73, 138)


# https://pinkwink.kr/1125?category=769346
# https://youbidan.tistory.com/19
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=samsjang&logNo=220517391218s