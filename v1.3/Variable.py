# -*- coding: utf-8 -*-
import tensorflow as tf
a = 3
# 创建变量: variable
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)     # 矩阵乘法
# 直接输出y, 发现输出的是tensorflow特有的对象类型,
# tensor格式,EG: Tensor("MatMul:0", shape=(1, 1), dtype=float32)
# 且没有经过初始化,直接调用y.eval(), 会抛出ValueError
# 即, 图的"骨架"已经构建好了,但是还没有值, 当完成了初始化操作后, w x就具有了具体的意义, 就可以运算了
print(y)
# print(y, y.eval())
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    print(y)
    print(y.eval())

"""最终输出
Tensor("MatMul:0", shape=(1, 1), dtype=float32)
[[ 2.]]
"""