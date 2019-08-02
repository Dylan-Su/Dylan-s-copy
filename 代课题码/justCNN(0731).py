#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:42:30 2019

@author: dylan
"""



'''
神经网络层数：四层卷积+四层池化
迭代次数：500次
优化器：梯度下降算法
准确率：42%
图片数量：3大类，270张。
优化器：梯度下降算法
优化器：梯度下降算法
'''
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from keras.utils import to_categorical

# 读取数据文件
names = locals()
batch_size = 15
train_steps = 1000
test_steps = 10

x = tf.placeholder(tf.float32,shape = (batch_size,1,384,256,3))
y = tf.placeholder(tf.int32,shape = (1,batch_size))
#x_image = tf.reshape(x, [1,2, 288, 432, 4])
#x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target))]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target


for i in range(310,580):
    #########图片路径
    m=i    
    m=str(m)   
    info=m+".jpg"
    #print(info) 
    #########字符串两端加“”双引号
    #info="\""+info+"\""
    #########读取图片  
    names['images_%s' % (m)]=tf.read_file(info, 'r')   
    #########图片解码
    names['image_tensor_%s' % (m)]=tf.image.decode_jpeg(names['images_%s' % (m)])

    #########读取图片形状
    names['shape_%s' % (m)] = tf.shape(names['image_tensor_%s' % (m)])
    #########形状操作
    names['session_%s' % (m)] = tf.Session()
    #print("图像的形状为：")
    #print((names['session_%s' % (m)]).run((names['shape_%s' % (m)])))
    ######### 将tensor转换为ndarray
    names['image_ndarray_%s' % (m)] = (names['image_tensor_%s' % (m)]).eval(session=(names['session_%s' % (m)]))
    # 显示图片
    #plt.imshow((names['image_ndarray_%s' % (m)] ))
    #plt.show()
    
    #########图片张量数据类型转换
    #names['image_tensor_%s' % (m)] = tf.cast((names['image_tensor_%s' % (m)]), tf.float32)
    names['image_tensor_%s' % (m)] = tf.cast((names['image_tensor_%s' % (m)]), tf.int32)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)  
    ########张量tensor变形----必须延展为5D
    for i in range(310,580):
        m=i    
        m=str(m)
        names['images_tensor_reshape_%s' % (m)]=sess.run(tf.reshape((names['image_tensor_%s' % (m)]),shape = [1,1,384,256,3]))
    #k4=tf.concat([images_tensor_reshape_0, images_tensor_reshape_1],1)
    k4=images_tensor_reshape_310
    x_shape4=tf.shape(k4) 
    print("合并的tensor2形状为：")
    print(sess.run(x_shape4))
    x_train = k4
    print(x_train)
    ########合并张量image_tensor_0+image_tensor_1；image_tensor_1+image_tensor_2；。。。。。。。。。image_tensor_21+image_tensor_22
    for i in range(311,580):
        m=i   
        #n=i+1
        m=str(m)
        #n=str(n)
        #names['x_train_concat_%s' % (m)]=tf.concat([(names['images_tensor_reshape_%s' % (m)]), (names['images_tensor_reshape_%s' % (n)])],1)
        #print(names['x_train_concat_%s' % (m)])
        x_train=tf.concat([x_train, (names['images_tensor_reshape_%s' % (m)])],0)
    x_shape5=tf.shape(x_train) 
    print("x_train形状为：")
    print(sess.run(x_shape5))
#sess.close()
    
###########以上面的 K4 作为 input数据，下面 设置 filter    
filter_1 = tf.Variable(tf.random_normal([1,1,1,3,1]))
filter_2 = tf.Variable(tf.random_normal([1,1,1,1,1]))
print(x_train)
print(type(filter))
y1 = np.zeros((1,90))
y2 = np.ones((1,90))
y3 = 2 * np.ones((1,90))
y1 = np.concatenate((y1,y2),axis = 1)
y1 = np.concatenate((y1,y3),axis = 1)
y1 = y1.reshape(1,270)
y1 = y1.astype(int)
y1 = y1[0]
print(y1)
print(y1.shape)
###########第一层卷积
op1 = tf.nn.conv3d(x, filter_1, strides=[1,1,1,1,1], padding='VALID')
###########第一层池化
pooling_3d_1=tf.nn.max_pool3d(op1,[1,1,2,2,1],[1,1,2,2,1],padding='VALID')
###########第二层卷积
op2 = tf.nn.conv3d(pooling_3d_1, filter_2, strides=[1,1,1,1,1], padding='VALID')
###########第二层池化
pooling_3d_2=tf.nn.max_pool3d(op2,[1,1,2,2,1],[1,1,2,2,1],padding='VALID')
###########第三层卷积
op3 = tf.nn.conv3d(pooling_3d_2, filter_2, strides=[1,1,1,1,1], padding='VALID')
###########第三层池化
pooling_3d_3=tf.nn.max_pool3d(op3,[1,1,2,2,1],[1,1,2,2,1],padding='VALID')
###########第四层卷积
op4 = tf.nn.conv3d(pooling_3d_3, filter_2, strides=[1,1,1,1,1], padding='VALID')
###########第四层池化
pooling_3d_4=tf.nn.max_pool3d(op4,[1,1,2,2,1],[1,1,2,2,1],padding='VALID')

flatten = tf.layers.flatten(pooling_3d_4)
print(flatten.shape)
print(pooling_3d_4.shape)
y_ = tf.layers.dense(flatten, 3)#这个数值是对应输出的标签的总数量，比如输入了10个测试数据，那就要输出10个预测值
sess = tf.Session()
x_shape6=tf.shape(y_) 
print("y_形状为：")
print(sess.run(x_shape6))
y_shape = tf.shape(y)
print("y形状为：")
print(sess.run(y_shape))


sess.close()
#y=tf.argmax(y, axis=1)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_,labels = y[0])
loss = tf.reduce_mean(loss,keep_dims = False)
print(loss.shape)
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y1[1],[-1]), logits=y_)
print(1)
# y_ -> sofmax
# y -> one_hot
# loss = ylogy_

# indices
predict = tf.argmax(y_, 1)
print(2)
#predict = y_
#pridict = np.array(predict)
# [1,0,1,1,1,0,0,0]
#y= tf.to_int64(y)
correct_prediction = tf.equal(tf.cast(predict,tf.int64), tf.cast(y,tf.int64))
print(3)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64),keep_dims = False)
print(accuracy.shape)
print(4)
#y= tf.to_float32(y)
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)#学习率和cost函数
 ###**************************************************************
#init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        print(1)
        
        #y=y.astype(np.float32)
        #y = tf.to_float32(y)
        batch_data, batch_labels = next_batch(x_train,y1,batch_size)
     #   batch_data, batch_labels = x_train.next_batch( batch_size )
        print(2)
        
     # 张量转化为ndarray
        #batch_data = session_2.run(batch_data)   
        print(3)
        sess.close()
        batch_data = tf.cast(batch_data,tf.int32)
        sess = tf.InteractiveSession()
        batch_data=sess.run(batch_data)
        print(4)
        batch_labels = np.array(batch_labels)
        batch_labels = batch_labels.reshape(1,batch_size)
        x_shape8=tf.shape(batch_labels) 
        print("x_train形状为：")
        print(sess.run(x_shape8))
        print(5)
        
       
        batch_data= np.array(batch_data)
       # batch_data=np.hsplit(batch_data,3)[0]
        print(6)
        
        
        #print(1)
        init = tf.global_variables_initializer()
        sess.run(init)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict = {
                        x: batch_data, 
                        y: batch_labels})
        print(loss_val.shape)
        print(acc_val.shape)
        if (i+1) % 10 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' 
                  % (i+1, loss_val, acc_val))
    sess.close()  

with tf.Session() as sess:        
    all_test_acc_val = []
    init = tf.global_variables_initializer()
    sess.run(init)
   
    x_shape11=tf.shape(x_train) 
    print("x_train形状为：")
    print(sess.run(x_shape11))
    for j in range(test_steps):
        test_batch_data, test_batch_labels = next_batch(x_train,y1,batch_size)
        
        
        ##################数据类型转换：必须把  test_batch_data 这样的tensor列表数据，转为矩阵（numpy）类型
        #test_batch_data= np.array(test_batch_data) 
        #sess = tf.InteractiveSession()
        test_batch_data = tf.to_int32(test_batch_data)
        test_batch_data=sess.run(test_batch_data)
     #   sess = tf.InteractiveSession()
        
        test_batch_labels = np.array(test_batch_labels)
        test_batch_labels = test_batch_labels.reshape(1,batch_size)
        x_shape8=tf.shape(test_batch_labels) 
        #print("x_train形状为：")
        #print(session_2.run(x_shape8))
        print(j)
        test_acc_val = sess.run(
            [accuracy],
            feed_dict = {
                x: test_batch_data, 
                y: test_batch_labels
            })
        all_test_acc_val.append(test_acc_val)
    test_acc = np.mean(all_test_acc_val)
    print('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc)) 
    sess.close()
print( "OK")
