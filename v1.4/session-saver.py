# -*- coding: utf-8 -*-
import tensorflow as tf
""" 模型保存, 使用 tf.train.Saver() """
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    '''全局初始化, 会保证计算图中 的所有变量都被初始化
    即: 整个计算图应该被顺序执行过, 这样所有Node的变量才能执行一遍'''
    sess.run(init_op)
    print(y.eval())
    # 模型分为3/4 (V2, V1)部分保存, 名字为 test.*
    save_path = saver.save(sess, "./model/test")
    print('Model saved in file:', save_path)