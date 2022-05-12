#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:17:42 2022

@author: yanyifan
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential


class BasicBlock(layers.Layer):
    #初始化函数
    #filter_num:理解为卷积核通道的数目，也就是channel的通道数
    #stride = 1意味着对图片不进行采样
    def __init__(self,filter_num,strides=1):
        #调用母类的初始化方法
        super(BasicBlock,self).__init__()
        #filter_num：卷积核通道的数目.(3,3):卷积核的size
        #padding='same'如果stride等于1，那么输出等于输入。
        #如果stride大于等于2的话，padding=same,会自动补全，
        # 如果等于2的话，输入是32x32,可能输出是14x14,那么如果padding=same
        #会padding输入的大小，使得输出是16x16


        self.conv1=layers.Conv2D(filter_num,(3,3),strides=strides,padding='same')
        self.bn1=layers.BatchNormalization()
        #非线性激活函数
        self.relu=layers.Activation('relu')

        #那么这里设置stride=1,就始终保持一样
        self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2=layers.BatchNormalization()

        if strides != 1:
            #下采样
            self.downsample=Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=strides))
        else:
            self.downsample=lambda x:x



    def call(self,inputs,training=None):
        #[b,h,w,c]
        out=self.conv1(inputs)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        identify=self.downsample(inputs)
        output=layers.add([out,identify])
        #使用tf的函数功能
        output=tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    def __init__(self,layer_dims,num_classes=100):
        #layer_dims:resnet18里面有[2,2,2,2]，也就是四个resblock
        #这里指定了一共有多少个resblock层，每个层有多少个basicblock
        #后面在设置blocks的数量的时候，就是用的这里的层的个数
      #一个resblock里面包含了两层basicblock
        #num_classes = 100:就是我们设置的输出的类的个数
        super(ResNet, self).__init__()

        #实现预处理层
        self.stem=Sequential([layers.Conv2D(64,(3,3),strides=(1,1)),
                              layers.BatchNormalization(),
                              layers.Activation('relu'),
                              #layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
                              ])
        #创建4个res_block
        #这里blocks的数量是layer_dims[0]
        #这里创建的四个res_block与前面的layer_dims:[2,2,2,2]对应
        #将stride设置为2是为了让feature_size越来越小
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2=self.build_resblock(128,layer_dims[1],strides=2)
        self.layer3=self.build_resblock(256,layer_dims[2],strides=2)
        self.layer4=self.build_resblock(512,layer_dims[3],strides=2)


        #out:[b,512,h,w]
        #经过运算之后不能得到h和w的值，
        #使用自适应的方法得到h,w
        #GlobalAveragePooling2D:就是不管你的长和宽是多少
        #会在某个channel上面的长和宽加起来，取均值
        self.avgpool=layers.GlobalAveragePooling2D()
        #创建全连接层
        #这里的Dense是用来分类的,这里输出是之前输出的类别，num_classes
        #self.flatten = layers.Flatten()
        self.fc=layers.Dense(num_classes,activation = 'softmax')



    def call(self,inputs,training=None):
        #完成前向运算过程
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #这里已经变成[b,c]的shape,不需要reshape了
        x=self.avgpool(x)
        #这里输出是[b,100]
        x=self.fc(x)

        return x



    def build_resblock(self,filter_num,blocks,strides=1):
        res_blocks=Sequential()
        #添加第一层basicblock
        #可能有下采样的功能的
        res_blocks.add(BasicBlock(filter_num,strides))
        #但是对于后面的basicblock不让有下采样功能
        #从1开始，一直到blocks个
        for _ in range(1,blocks):
            #这样只会在第一个下采样，后面的不在下采样，保持shape不变
            res_blocks.add(BasicBlock(filter_num,strides=1))
        return res_blocks

def resnet18():
    return ResNet([2,2,2,2])

def resnet34():

    return ResNet([3, 4, 6, 3])  #4个Res Block，第1个包含3个Basic Block,第2为4，第3为6，第4为3



