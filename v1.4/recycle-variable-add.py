# -*- coding: utf-8 -*-
import tensorflow as tf
"""
一个小栗子,循环加一,并输出每次的结果
"""

state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        # 两种方式均可输出state值
        # 1. sess.run() 返回值 2. *.eval()
        print(sess.run(state))
        # print(state.eval())
        sess.run(update)
    print(sess.run(state))

