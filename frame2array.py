#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:13:19 2019

@author: dylan
"""

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib

cap = cv.VideoCapture('/Users/dylan/keyframe_out1.mov')
if cap.isOpened():
    print('success!')
    exit(1)
else:
    print('Can\'t find the keyframe')
count = 0
a = 1
i = 0
ret,key_frame = cap.read()#逐帧读取关键帧图像,读取第一帧图像
gray_key_frame = cv.cvtColor(key_frame,cv.COLOR_BGR2GRAY)#将第一帧图像转换为灰度图像
data = np.asarray(gray_key_frame)#将第一帧图像转换成数组的形式
data = data.reshape(1080,1920,1)
dim = data.shape
print(dim)
print(gray_key_frame)
print(data.shape)
print('first data')
while ret:
    gray_key_frame = cv.cvtColor(key_frame,cv.COLOR_BGR2GRAY)#将每帧图像转为灰度
    img = np.asarray(gray_key_frame)#将每帧图像变成数组的形式进行存储
    #'''
    data = np.append(data,img)#将新读取的帧图像像素拼接到之前的数组中
    data = data.reshape(dim[0],dim[1],dim[2]+i+1)#对拼接好的数组重新塑造维度
    print(data)
    print(data.shape)
    #'''
    '''先转换成列表，使用append函数进行列表的拼接，然后再转回数组的格式（报错）
    data_list = data.tolist
    img_list = img.tolist
    data_list.append(img_list)
    data = np.array(data_list)
    '''
    
    #print(img)
    #dim = img_copy.shape
    #data1 = data.shape((dim[0]),(dim[1]),(dim[2]+1))
    count = count + 1
    i = i + 1
    print("this is",i,"th out")
    ret,key_frame = cap.read()
print(data)
print(count)