"""

@file  : 005-optimizer.py

@author: xiaolu

@time  : 2019-07-10

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)
np.random.seed(1)

LR = 0.01   # 指定学习率
BATCH_SIZE = 32  # 批量的大小

# 制造一批假的数据
x = np.linspace(-1, 1, 100).reshape((100, 1))
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

# plt.scatter(x, y)
# plt.show()


# 定义网络结构
class Net:
    # 此处opt传进来的是不同的优化器
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        l = tf.layers.dense(self.x, 20, tf.nn.relu)   # 定义一个dense层
        out = tf.layers.dense(l, 1)
        self.loss = tf.losses.mean_squared_error(self.y, out)
        self.train = opt(LR, **kwargs).minimize(self.loss)


net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_his = [[], [], [], []]   # 记录损失

for step in range(300):
    index = np.random.randint(0, x.shape[0], BATCH_SIZE)   # 生成批量化的数据的索引
    b_x = x[index]
    b_y = y[index]

    for net, l_his in zip(nets, losses_his):
        _, l = sess.run([net.train, net.loss], {net.x: b_x, net.y: b_y})
        l_his.append(l)     # loss recoder


labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
