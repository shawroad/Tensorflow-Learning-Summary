"""

@file  : 017-Maxout网络实现mnist分类.py

@author: xiaolu

@time  : 2019-10-17

"""
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)  # (60000, 28, 28)
# print(x_test.shape)  # (10000, 28, 28)
# print(y_train.shape)  # (60000, )

x_train = x_train.reshape(-1, 28*28) / 255.
x_test = x_test.reshape(-1, 28*28) / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)  # (60000, 784)
print(y_train.shape)  # (60000, 10)


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:
        axis = -1

    num_channels = shape[axis]  # 获取通道数
    # if num_channels % num_units:
    #     raise ValueError('number of features({}) is not '
    #                      'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    print(shape)  # [-1, 50, 2]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None, 10])

W = tf.Variable(tf.random_normal([784, 100]))
b = tf.Variable(tf.random_normal([100, ]))
z = tf.matmul(x, W) + b

# #maxout = tf.reduce_max(z,axis= 1,keep_dims=True)
#
maxout= max_out(z, 50)

W2 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
b2 = tf.Variable(tf.random_normal([10, ]))

pred = tf.matmul(maxout, W2) + b2
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

learning_rate = 0.04
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 200
batch_size = 128
display_step = 1

# 启动sess
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(0, (x_train.shape[0] - batch_size), batch_size):
            batch_x, batch_y = x_train[i: i + batch_size], y_train[i: i + batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c

            # print('epoch:%d, step: %d, loss: %f'%(epoch, i // batch_size, c))
        avg_cost /= (x_train.shape[0] // batch_size)
        print('epoch: %d, loss: %f' % (epoch, avg_cost))
