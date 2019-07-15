"""

@file  : 014-save_load_model.py

@author: xiaolu

@time  : 2019-07-15

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
# print(train_X.shape)   # (100,)
train_Y = 2 * train_X + np.random.randn(train_X.shape[0]) * 0.3   # y=2x，但是加入了噪声

# 显示模拟数据点
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()

# 重置图
tf.reset_default_graph()

# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W) + b

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))   # 均方误差
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# 初始化变量
init = tf.global_variables_initializer()

# 训练参数
training_epochs = 20
display_step = 2

# 保存模型
saver = tf.train.Saver()
saved_dir = './log/'


plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    # a传进了的是一个损失列表　　也就是每次的损失　我们不想把每步的损失画出来, 我们只想每十步进行
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# # 启动session
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Fit all training data
#     for epoch in range(training_epochs):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#
#         # 显示训练中的详细信息
#         if epoch % display_step == 0:
#             loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
#             print("Epoch:{}, cost:{}, W:{}, b:{}".format(epoch+1, loss, sess.run(W), sess.run(b)))
#
#             if not (loss == "NA"):  # 代表此次的损失不是数字　是噪音　过滤掉
#                 plotdata["batchsize"].append(epoch)
#                 plotdata["loss"].append(loss)
#
#     print(" Finished!")
#
#     # 用saver去保存模型
#     saver.save(sess, saved_dir+'linearmodel.cpkt')
#
#     # 训练结束　打印一下损失和学习到的参数
#     print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
#
#     # # 图形显示
#     # plt.plot(train_X, train_Y, 'ro', label='Original data')
#     # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#     # plt.legend()
#     # plt.show()
#     # # 损失
#     # plotdata["avgloss"] = moving_average(plotdata["loss"])
#     # plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
#     # plt.xlabel('Minibatch number')
#     # plt.ylabel('Loss')
#     # plt.title('Minibatch run vs. Training loss')
#     # plt.show()
#
#     print("x=0.2, z=", sess.run(z, feed_dict={X: 0.2}))


with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, saved_dir+'linearmodel.cpkt')
    print('x=0.2, z=', sess2.run(z, feed_dict={X: 0.2}))





