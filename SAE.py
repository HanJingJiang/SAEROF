import os           # https://blog.csdn.net/marsjhao/article/details/73480859    https://blog.keras.io/building-autoencoders-in-keras.html
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# 读数据
SampleFeature = []
ReadMyCsv(SampleFeature, "SampleFeature.csv")
SampleFeature = np.array(SampleFeature)
print('SampleFeature',len(SampleFeature))
print('SampleFeature[0]',len(SampleFeature[0]))
x = SampleFeature #
# from sklearn.cross_validation import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, x, test_size=0.2)    # 切分数据集进行训练，用全部数据集x进行“预测”！！！！

x_train = SampleFeature  # (3530, 1209)
x_test = SampleFeature
# 改变数据类型
x_train = x_train.astype('float32') / 1.
x_test = x_test.astype('float32') / 1.
print(x_train.shape)        # (3866, 906)
print(x_test.shape)         # (3866, 906)
print(type(x_train[0][0]))        # <class 'numpy.float32'>

# 变量
encoding_dim = 100
input_img = Input(shape=(len(SampleFeature[0]),))    # 输入维度

# 构建autoencoder
from keras import regularizers
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-7))(input_img)    # 与单层的唯一区别 (from keras import regularizers)!!!注意调节参数10e-7
decoded = Dense(906, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=50, shuffle=True, validation_data=(x_test, x_test))

# 预测
encoded_imgs = encoder.predict(x)
decoded_imgs = decoder.predict(encoded_imgs)
print(len(encoded_imgs))  #3866
print(len(encoded_imgs[1])) #100
storFile(encoded_imgs, '100SampleFeature.csv')
