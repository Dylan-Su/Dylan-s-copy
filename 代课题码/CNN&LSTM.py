#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:10:03 2019

@author: dylan
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Reshape
#使用sequential
model = Sequential()
#先构建VGGnet


#构建浅层VGG卷积神经网络
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),name = "MaxPooling_1"))
model.get_layer(index=3).output_shape
'''
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))    #此时输出的是64个feature_map,每个feature_map的大小为32*32
model.add(MaxPooling2D(pool_size=(2, 2)))   #输出32*32的图像特征
'''
#问题：将CNN池化层处理过的图像直接输入到LSTM网络中会出现报错：
#   ‘’Input 0 is incompatible with layer lstm_2: expected ndim=3, found ndim=4‘’
#   解决办法：可以先读取池化层的输出图像，观察其输出维度，然后用tf.reshape进行更改

#将池化层处理后的图像输出表示出来
#Flatten是numpy下的一个函数，即返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的。
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.get_layer(index=4).output_shape
model.add(Reshape((576,576)))
#model.get_layer(index=5).output_shape
#构建LSTM网络
nb_lstm_outputs = 30  #神经元个数
nb_time_steps = 1024  #时间序列长度
nb_input_vector = 32 #输入序列


model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector),return_sequences=True))
model.get_layer(index=6).output_shape


model.add(Flatten())
model.add(Dense(5, activation='softmax'))

#对上述构造的模型进行编译，loss表示损失函数，
model.compile(loss='categorical_crossentropy',                                 # matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',#优化器选项，预定义一个优化器。该参数可指定为已预定义的优化器名
              metrics=['accuracy'])#性能评估选项，如accuracy是准确率这个评估参数。
                                   #提供了一系列用于模型性能评估的函数,这些函数在模型编译时由metrics关键字设置。
#对训练集中的数据进行图像增广
train_datagen = ImageDataGenerator(
        rescale=1./255,   #值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，
                          #这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
        shear_range=0.2,  #浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
        zoom_range=0.2,   #若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大。
        horizontal_flip=True)  #布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。

#对测试集集中的数据进行图像增广
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
        '/Users/dylan/anaconda3/Data/keras/re/train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/Users/dylan/anaconda3/Data/keras/re/validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

#使生成器与模型并行执行，以提高效率
model.fit_generator(
        train_generator,                      #生成器函数名称
        samples_per_epoch=200,               #数值为整数，当生成器返回2000次数据时计一个epoch结束，执行下一个epoch
        nb_epoch=20,                          #数据迭代的轮数
        validation_data=validation_generator, #交叉验证集的生成器
        nb_val_samples=800)          
