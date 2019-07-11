"""

@file  : 002-SimpleRegression.py

@author: xiaolu

@time  : 2019-07-10

"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 100).reshape((100, 1))    # shape=(100, 1)

noise = np.random.normal(0, 0.1, size=x.shape)  # 给定均值和方差按正太分布随机初始化
y = np.power(x, 2) + noise    # shape (100, 1) + some noise   式子 x**2 + noise

# plt.scatter(x, y)
# plt.show()
# exit()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y


l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # 计算损失
optimizer = tf.train.AdamOptimizer()
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting

for step in range(1000):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
