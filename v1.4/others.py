# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
'''其它, 补充部分
1. numpy 转 tf 张量
2. tf.placeholder 类型
'''
a = np.zeros((3, 3))
ta = tf.convert_to_tensor(a)

tc = tf.constant(1)
tv = tf.Variable([2, 3])
with tf.Session() as sess:
    # 1. tf.constant 可以不经过初始化,直接输出其value
    print(sess.run(tc))
    # 2. tf.variable 不经过初始化,会报错
    # print(sess.run(tv))

    # 1.2.两步的结果说明
    # 经过tf.conver_to_tensor转换的结果,应该为tf.constant类型
    print(sess.run(ta))

'''
1. placeholder类型的使用
2. Variable + placeholder组成的计算图 运行顺序的问题?
'''
# var1 = tf.Variable([4.])
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# 注: 版本升级到1.0之后, -*名字变换, 除了 +/add div
# output = tf.multiply(input1, input2) # 7*2=14
output = tf.divide(input1, input2) # tf.div 7/2=3.5
# output = tf.divide(input1, var1) # >>tf.div 7/6=1.75<<
# output = tf.negative(input1) # -(7) = 7
# output = tf.subtract(input1, input2) # 7-2=5

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print(sess.run(output, feed_dict={input1: [7.0], input2: [2.0]}))