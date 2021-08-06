'''
# 경고제어
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''

import sys
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 12

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

n = 0
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

print('The Answer is ', model.predict_classes(x_test[n].reshape((1, 28, 28, 1))))

model.save('mnist(CNN).hdf5') 
'''
model = tf.keras.models.load_model('mnist(CNN).hdf5')

import random

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=1)

wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)

count = 0
nrows = ncols = 4

plt.figure(figsize=(12,8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()


# 직접 작성한 글 확인
# OpenCv 사용
# pip install opencv-python
import cv2

# model = tf.keras.models.load_model('mnist(CNN).hdf5')
for i in range(10):
    my_img = cv2.imread('images/num'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(my_img)
    # plt.show()
    
    my_img = cv2.resize(255-my_img, (28, 28))
    test_my_img = my_img.flatten() / 255.0
    #print(test_my_img)
    # 배경이 0으로 되어있는 모델이라 변경
    test_my_img = np.where(test_my_img == 1, 2, test_my_img)
    test_my_img = np.where(test_my_img == 0, 1, test_my_img)
    test_my_img = np.where(test_my_img == 2, 0, test_my_img)
    #print(test_my_img)
    
    test_my_img = test_my_img.reshape((-1, 28, 28, 1))
    print(test_my_img)
    print('The Answer is ', model.predict_classes(test_my_img))





# https://pinkwink.kr/1121