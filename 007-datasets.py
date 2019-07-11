"""

@file  : 007-datasets.py

@author: xiaolu

@time  : 2019-07-11

"""
import tensorflow as tf
import numpy as np


# random create data
data_x = np.random.uniform(-1, 1, (1000, 1))
data_y = np.power(data_x, 2) + np.random.normal(0, 0.1, size=data_x.shape)

# split train data and test data.　Then, we will split at point 800
x_train, x_test = np.split(data_x, [800])
y_train, y_test = np.split(data_y, [800])


tfx = tf.placeholder(x_train.dtype, x_train.shape)
tfy = tf.placeholder(y_train.dtype, y_train.shape)

# create dataloader
dataset = tf.data.Dataset.from_tensor_slices(((tfx, tfy)))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.repeat(3)    # 3个epoch
iterator = dataset.make_initializable_iterator()  # 初始化这个迭代器

# define network
bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, data_y.shape[1])

# define loss and optimizer
loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 制作批数据
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: x_train, tfy: y_train})

for step in range(201):
    try:
        _, train_loss = sess.run([train, loss])
        if step % 10 == 0:
            test_loss = sess.run(loss, {bx: x_test, by: y_test})   # 注意这里的测试集直接
            print('step: %i/200' % step, '|train loss:', train_loss, '|test loss:', test_loss)
    except tf.errors.OutOfRangeError:
        print('Finish the last epoch')



