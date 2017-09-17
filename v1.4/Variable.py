# -*- coding: utf-8 -*-
import tensorflow as tf
"""tf各种变量,包括variable, constant数据类型的参数
在经过最后一步的初始化之前,调用*.eval()输出参数内容都会出ValueError错
"""
# tf操作类似numpy
# 常量矩阵, 推荐使用float32
tf.zeros([3, 4], tf.int32)

tensor = [[1, 2, 3], [4, 5, 6]]
# 同样尺寸大小的新tf矩阵, 内容全部为0
# tensor为[[1, 2, 3], [4, 5, 6]]
# 则,下面输出为 同尺寸大小的全0数组 [[0, 0, 0], [0, 0, 0]]
tf.zeros_like(tensor)

# 类似zeros函数, ones函数操作
# [[1, 1, 1], [1, 1, 1]]
tf.ones([2, 3], tf.int32)

tf.ones_like(tensor)

# 常量1-D, 直接输出 其值 依然会报错
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
# print tensor.eval()

# 常量2-D, 输出 [[-1, -1, -1], [-1, -1, -1]]
tf.constant(-1.0, shape=[2, 3])

# 线型等分, 3表示分成3分[10.0, 11.0, 12.0]
tensor = tf.linspace(10.0, 12.0, 3, name='linspace')
# tensor = tf.linspace(10.0, 12.0, 3, name='linspace')
# print tensor

# start:3, limit: 18, delta: 3, 含头不含尾
# 则输出结果: [3, 6, 9, 12, 15]
start, limit, delta = 3, 18, 3
ran = tf.range(start, limit, delta)

# with tf.Session() as sess:
#     sess.run(ran)
#     print(ran.eval())

"""
tensorflow 随机值
"""
# 服从高斯(正态)分布, 均值为-1, 方差为4
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# 洗牌操作,仅针对第一维度
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

with tf.Session() as sess:
    sess.run(norm)
    sess.run(shuff)
    print(norm.eval())
    print(shuff.eval())

"""输出
[[-2.96248388  7.86170959  0.65325689]
 [-6.00747633 -1.06459153  1.50358748]]
[[3 4]
 [1 2]
 [5 6]]
"""