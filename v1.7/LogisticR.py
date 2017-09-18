# -*- coding: utf-8 -*-
"""
简单逻辑回归实验: 本质还是分类任务
1. 第一步, input_data 下载并解析mnist数据集, 属性
    * mnist.train.images
    * mnist.train.labels
    * mnist.test.images
    * mnist.test.labels

"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集数据加载
mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print('MNIST loaded')

# 查看下对象属性
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)
# 元素所在位置为1 [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
print(trainlabel[0])