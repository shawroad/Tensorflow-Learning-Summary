"""

@file  : 016-tensorboard-use.py

@author: xiaolu

@time  : 2019-07-15

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 看一下tensorflow的版本
print("TensorFlow 版本: {}".format(tf.VERSION))


# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3  # y=2x，但是加入了噪声

# 图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

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

# 将预测值用直方图画出
tf.summary.histogram('z', z)

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))

# 将损失以标量显示
tf.summary.scalar('loss_function', cost)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
# 参数设置
training_epochs = 20
display_step = 2

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# 启动session
with tf.Session() as sess:
    sess.run(init)

    merged_summary_op = tf.summary.merge_all()  # 合并所有summary　　常规写法　就是将你上面所要收集的东西全部合并起来

    # 创建summary_writer，用于写文件
    summary_writer = tf.summary.FileWriter('log/summaries', sess.graph)  # 将当前这个图先写入

    # begin fit
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})  # 将收集的东西计算出
            summary_writer.add_summary(summary_str, epoch)   # 将summary 写入文件

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))

            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print(" Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    # print ("cost:",cost.eval({X: train_X, Y: train_Y}))

    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()

    print("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))


# tensorboard --logdir /summaries
