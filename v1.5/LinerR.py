# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

"""线性回归模型"""
# 随机生成1000个点, 围绕在 y=0.1x+0.3直线周围
num_points = 1000
vectors_set = []

# 均值0.0, 标准差0.55
x1 = np.random.normal(0.0, 0.55, num_points)
y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03, num_points)

# 生成的样本, 是否转换成python特有对象类型(eg: list)均可满足下面的绘图
x_data = x1 #list(x1)
y_data = y1 #list(y1)

# 散点图, 颜色设置为red
plt.scatter(x_data, y_data, c='r')
plt.savefig('LineR.png')
# plt.show()

# 生成1维的W矩阵, 取值是[-1, 1]之间的随机数
W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='W')
# 生成1维的b矩阵, 初始值是0
b = tf.Variable(tf.zeros(shape=[1]), name='b')
# 经过计算得到预估值y
y = W*x_data + b

# 以预估值y和实际值y_data之间的均方误差作为损失,
# tf.reduce_mean作用: 先求和, 结果再取平均值
tmp = tf.square(y - y_data)
loss = tf.reduce_mean(tmp, name='loss')
#
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#
train = optimizer.minimize(loss=loss, name='train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 初始化的W和b是多少?
    print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))
    # print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss), sum(sess.run(tmp)))
    # 执行训练过程, 共20次训练
    for _ in range(20):
        sess.run(train)
        print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))



