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



def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return



SampleFeature = []
ReadMyCsv(SampleFeature, "SampleFeature.csv")
SampleFeature = np.array(SampleFeature)
print('SampleFeature',len(SampleFeature))
print('SampleFeature[0]',len(SampleFeature[0]))
x = SampleFeature 

x_train = SampleFeature  
x_test = SampleFeature
x_train = x_train.astype('float32') / 1.
x_test = x_test.astype('float32') / 1.
print(x_train.shape)      
print(x_test.shape)        
print(type(x_train[0][0]))       

encoding_dim = 100
input_img = Input(shape=(len(SampleFeature[0]),))    


from keras import regularizers
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-7))(input_img)    
decoded = Dense(906, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=50, shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x)
decoded_imgs = decoder.predict(encoded_imgs)
print(len(encoded_imgs)) 
print(len(encoded_imgs[1])) 
storFile(encoded_imgs, '100SampleFeature.csv')
