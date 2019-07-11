"""

@file  : 006-tensorboard使用.py

@author: xiaolu

@time  : 2019-07-11

"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# 1. make fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary
    # 在训练神经网络时，当需要查看一个张量在训练过程中值的分布情况时，可通过
    # tf.summary.histogram()将其分布情况以直方图的形式在TensorBoard直方图仪表板上显示
    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# plot scalar variation with times
tf.summary.scalar('loss', loss)     # add loss to scalar summary

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all()                       # operation to merge all summary

for step in range(100):
    _, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
    writer.add_summary(result, step)

# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# http://localhost:6006
