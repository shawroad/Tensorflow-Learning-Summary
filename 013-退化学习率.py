"""

@file  : 013-退化学习率.py

@author: xiaolu

@time  : 2019-07-17

"""
import tensorflow as tf


global_step = tf.Variable(0, trainable=False)

# 初始化学习率
initial_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps=10,  # 每个10步退化一次
                                           decay_rate=0.9)

opt = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)  # 定义一个op 记步, 每次让global_step + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 打印学习率　看学习率的变化
    print(sess.run(learning_rate))
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        print("当前第{}步, 学习率为:{}".format(g, rate))

