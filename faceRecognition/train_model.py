from dataSet import DataSet
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Conv2D,BatchNormalization
from keras.layers import MaxPooling2D,Flatten,Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "F:\GraduationProject\model.h5"   #模型进行存储和读取的地方
    IMAGE_SIZE = 128    #模型接受的人脸图片大小128*128

    def __init__(self):
        self.model = None

    #读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset

        """
        CNN模型:
        卷积 -> 批归一化 -> 激活 ->  池化

        卷积 -> 批归一化 -> 激活 ->  池化 -> dropout
        卷积 -> 批归一化 -> 激活 ->  池化 -> dropout

        展平 -> 全连接 -> 批归一化 -> 激活 -> dropout
            -> 全连接 -> 批归一化 ->  激活 -> dropout
            -> 全连接 -> 批归一化 ->  激活 -> summary
        """


    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(

                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='tf',
                input_shape=self.dataset.X_train.shape[1:],

            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.15))

        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.15))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

        """
        进行模型训练的函数
            optimizer 可选: RMSprop,Adagrad,adadelta
            loss：可选：categorical_crossentropy  squared_hinge
        """

    def train_model(self):

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=8, batch_size=100)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print('test loss;', loss)
        print('test accuracy:', accuracy)


    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    #需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self, img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)  #测算一下该img属于某个label的概率
        max_index = np.argmax(result) #找出概率最高的
        return max_index, result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率


if __name__ == '__main__':
    datast = DataSet('F:\GraduationProject\pictures\dataset')
    model = Model()
    model.read_trainData(datast)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()













