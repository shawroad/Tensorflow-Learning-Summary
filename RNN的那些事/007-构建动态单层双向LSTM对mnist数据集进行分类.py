"""

@file  : 007-构建动态单层双向LSTM对mnist数据集进行分类.py

@author: xiaolu

@time  : 2019-07-12

"""
from keras.datasets import mnist
import tensorflow as tf
from keras.utils import to_categorical

max_feature = 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y = tf.placeholder(tf.float32, shape=[None, 10])

n_hidden = 64
learning_rate = 0.001
training_iters = 60000
batch_size = 128
display_step = 10

# 正向LSTM
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# 反向LSTM
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# 其实下面是多层的双向动态的实现,我们只是堆叠一层而已
outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([lstm_fw_cell], [lstm_bw_cell], x, dtype=tf.float32)

# print(outputs[0].shape, len(outputs))  # (?, 128)  28  # 默认是将两个方向的输出拼接起来

outputs = tf.transpose(outputs, [1, 0, 2])

# 我们只取最后一层的输出,将其拼接起来
pred = tf.contrib.layers.fully_connected(outputs[-1], 10)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# model evaluate
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = x_train[batch_size*(step-1): batch_size*step], y_train[batch_size*(step-1): batch_size*step]
        batch_x = batch_x.reshape((batch_size, 28, 28))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:    # 每隔10步显示一次
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

            print('Iter:' + str(step * batch_size) + ', Minbatch_loss: ' + str(loss)+ ', Training Accuracy: '+ str(acc))

        step += 1
    print(" Finished!")





