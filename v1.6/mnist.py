# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print('packs loaded')

print('Download and Extract MNIST dataset')
mnist = input_data.read_data_sets('data/', one_hot=True)
print
print('type of "mnist" is %s' % (type(mnist)))
print('number of train data is %d' % (mnist.train.num_examples))
print('number of test data is %d' % (mnist.test.num_examples))

# what's the data of MNIST look like?
print('What does the data of MNIST look like?')
trainimg    = mnist.train.images
trainlabel  = mnist.train.labels
testimg     = mnist.test.images
testlabel   = mnist.test.labels
# 训练数据 55000 784=28*28
# 测试数据 10000 784=28*28
# 10, 十分类
print
print('type of trainimg is %s' % (type(trainimg)))
print('type of trainlabel is %s' % (type(trainlabel)))
print('type of testimg is %s' % (type(testimg)))
print('type of testlabel is %s' % (type(testlabel)))
print('shape of trainimg is %s' % (trainimg.shape,))
print('shape of trainlabel is %s' % (trainlabel.shape,))
print('shape of testimg is %s' % (testimg.shape,))
print('shape of testlabel is %s' % (testlabel.shape,))

# 查看下 train数据集中的数据, 使用matplotlib绘图
nsample = 5
# 随机数范围[0, 55000)
randidx = np.random.randint(trainimg.shape[0], size=nsample)
for i in randidx:
    # a image 28*28
    curr_img    = np.reshape(trainimg[i, :], (28, 28))
    curr_label  = np.argmax(trainlabel[i, :]) # 最大值,也就是1, 所在位置的index下标

    plt.figure()
    plt.imshow(curr_img, cmap='gray')
    plt.title(''+str(i)+'th Training Data' + \
              'Label is ' + str(curr_label))
    print('' + str(i) + 'th Training Data')
    plt.axis('off')
    plt.savefig('mnist_'+str(i)+'.png')
# 决定最终是否显示(我们是否能够看到图像)
# plt.show()

# Batch Learning?
#
print('Batch Learning')
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
print('type of "batch_xs" is %s' % (type(batch_xs)))
print('type of "batch_ys" is %s' % (type(batch_ys)))
print('shape of "batch_xs" is %s' % (batch_xs.shape,))
print('shape of "batch_ys" is %s' % (batch_ys.shape,))


