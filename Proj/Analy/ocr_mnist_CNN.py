from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


img_rows = 28
img_cols = 28
batch_size = 128
num_classes = 10
epochs = 12
input_shape = (img_rows, img_cols, 1)

def main():
    # MNIST 데이터 읽어 들이기
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    input_shape = (img_rows, img_cols, 1)
    # 데이터 정규화
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    # 모델 구축
    model = build_model()
    model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(x_test, y_test))
    # 모델 저장
    model.save_weights('mnist_CNN.hdf5')
    # 모델 평가
    score = model.evaluate(x_test, y_test, verbose=0)
    print('score=', score)


def build_model():
    # MLP 모델 구축
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    main()
