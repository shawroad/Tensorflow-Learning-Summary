"""

@file  : 009-构建多层动态双向RNN对mnist数据集进行分类.py

@author: xiaolu

@time  : 2019-07-12

"""
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical


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

stack_f_lstm = []
stack_b_lstm = []
for i in range(3):
    stack_f_lstm.append(tf.contrib.rnn.LSTMCell(n_hidden))
    stack_b_lstm.append(tf.contrib.rnn.LSTMCell(n_hidden))

mcell = tf.contrib.rnn.MultiRNNCell(stack_f_lstm)
mcell_bw = tf.contrib.rnn.MultiRNNCell(stack_b_lstm)

outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([mcell], [mcell_bw], x, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
print(outputs[0].shape, outputs.shape)

pred = tf.contrib.layers.fully_connected(outputs[-1], 10, activation_fn=None)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
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



