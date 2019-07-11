"""

@file  : 008-CNN.py

@author: xiaolu

@time  : 2019-07-11

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

tf.set_random_seed(1)
np.random.seed(1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)   # (60000, 28, 28)
print(x_test.shape)   # (60000, 28, 28)

x_train = x_train.astype('float32').reshape((-1, 28, 28, 1)) / 255.
x_test = x_test.astype('float32').reshape((-1, 28, 28, 1)) / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

gen_x = tf.placeholder(tf.float32, shape=x_train.shape)
gen_y = tf.placeholder(tf.int32, shape=y_train.shape)

# 制造批数据
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=x_train.shape[0])
dataset = dataset.batch(128)
dataset = dataset.repeat(1)   # 相当于训练三个epoch  把数据进行扩充三遍
iterator = dataset.make_initializable_iterator()  # 到此, 这个迭代器做好了　


# X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# Y = tf.placeholder(tf.float32, shape=[None, 10])

# CNN
X, Y = iterator.get_next()
conv1 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)   # 折半　shape=(14, 14, 16)
conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)   # 再折半 shape=(7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32])
output = tf.layers.dense(flat, 10)

# define loss, optimizer, and accuracy
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=output)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
# return (acc, update_op), and create 2 local variables

sess = tf.Session()
# sess.run([iterator.initializer], feed_dict={gen_x: x_train, gen_y: y_train})
# # the local var is for accuracy_op
# init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)     # initialize var in graph
sess.run([iterator.initializer, tf.global_variables_initializer(), tf.local_variables_initializer()],
         feed_dict={gen_x: x_train, gen_y: y_train})

for step in range(500):    # 每一步数据128
    try:
        _, pred, acc = sess.run([train_op, output, accuracy])

        if step % 10 == 0:
            print("step:{}, train_acc:{}".format(step, acc))
    except tf.errors.OutOfRangeError:
        print('Finish the laster epoch')






