# 파이값으로 이미지 읽어오고 모델 만들어 분석해보기
# https://archive.org/stream/Pi_to_100000000_places/pi.txt
import sys
import numpy as np
import cv2
import tensorflow as tf   



# 이미지 읽어오기 
filename = 'pi500'
img = cv2.imread("images/"+filename+".PNG")

# 그레이스케일로 변환하고 블러걸고 이진화하기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 그레이스케일 변환
blur = cv2.GaussianBlur(gray, (5, 5), 0)    # 블러
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)  # 2진화

# 윤곽 추출 -> 인덱스를 [0]으로 해줘야함 또는 두개의 변수 기입
# contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


'''
# 추출한 윤곽 반복처리
for i, each in enumerate(contours):
    x, y, w, h = cv2.boundingRect(each)
    if h < 18 or w < 1 : continue   # 너무 작으면 건너뛰기 
    if w > 50 or h > 50 : continue  # 너무 크면 건너뛰기
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), h, 2)
    
    
# 이미지 저장    
cv2.imwrite("images(box)/"+filename+"-ocr.png", img)    
'''


rects = []
img_w = img.shape[1]
# 추출한 좌표 처리
for i, each in enumerate(contours):
    x, y, w, h = cv2.boundingRect(each)
    if h < 18: continue   # 너무 작으면 건너뛰기 
    # if w > 50 or h > 50 : continue  # 너무 크면 건너뛰기
    red = (0, 0, 255)
    y2 = round(y / 10) * 10  # Y좌표 맞추기
    index = y2 * img_w + x
    rects.append((index, x, y, w, h))
rects = sorted(rects, key=lambda x: x[0])  # 정렬하기


# 해당 영역 이미지 데이터 추출
X = []
for i, r in enumerate(rects):
    index, x, y, w ,h = r
    num = gray[y:y+h, x:x+w]  # 부분 이미지 추출하기
    num = 255 - num  # 반전하기
    # 정사각형 내부에 그림 옮기기
    ww = round((w if w > h else h) * 1.85) 
    spc = np.zeros((ww, ww))
    wy = (ww-h)//2
    wx = (ww-w)//2
    spc[wy:wy+h, wx:wx+w] = num
    num = cv2.resize(spc, (28, 28))  # MNIST로 분석한 모델로 분석하기 위해 크기조절
    # cv2.imwrite("images(box)/"+filename+"-"+str(i)+".png", num)    # 이미지 저장
    # 데이터 정규화
    num = num.reshape(28*28)
    num = num.astype("float32") / 255
    X.append(num)
    
# 예측하기 
s = "14159265358979323846264338327950288419716939937510" + \
"58209749445923078164062862089986280348253421170679" + \
"82148086513282306647093844609550582231725359408128" + \
"48111745028410270193852110555964462294895493038196" + \
"44288109756659334461284756482337867831652712019091" + \
"45648566923460348610454326648213393607260249141273" + \
"72458700660631558817488152092096282925409171536436" + \
"78925903600113305305488204665213841469519415116094" + \
"33057270365759591953092186117381932611793105118548" + \
"07446237996274956735188575272489122793818301194912" 
answer = list(s)
from Analy import ocr_mnist_CNN
mnist = ocr_mnist_CNN.build_model()
mnist.load_weights('mnist(CNN).hdf5')

cnt = 0
nlist = mnist.predict(np.array(X))
for i, n in enumerate(nlist):
    ans = n.argmax()
    if ans == int(answer[i]):
        cnt += 1
    else:
        print("[ng]",i,"번째", ans, "!=", answer[i], np.int32(n*100))
        
print("정답률 : ", cnt/len(nlist))

