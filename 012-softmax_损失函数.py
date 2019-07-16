"""

@file  : 012-softmax_损失函数.py

@author: xiaolu

@time  : 2019-07-16

"""
import tensorflow as tf

# labels:假设是一组标签,用one_hot编码表示　　　logits: 假设这是模型最后的输出
labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]


logits_scaled = tf.nn.softmax(logits)   # 对输出进行两次softmax  这是第一次
logits_scaled2 = tf.nn.softmax(logits_scaled)    # 这是第二次

# 正确的交叉损失熵
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

# 记住, 往softmax_cross_entropy_with_logits() 不需要提前对预测结果进行softmax  这个函数会自己进行softmax
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

# 自己定义交叉上的样子
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled=", sess.run(logits_scaled))
    print("scaled2=", sess.run(logits_scaled2))  # 经过第二次的softmax后，分布概率会有变化

    print("rel1=", sess.run(result1), "\n")  # 正确的方式
    print("rel2=", sess.run(result2), "\n")  # 如果将softmax变换完的值放进去会，就相当于算第二次softmax的loss，所以会出错
    print("rel3=", sess.run(result3), "\n")


# 标签总概率为1
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel4=", sess.run(result4), "\n")


# 重点
# sparse
labels = [2, 1]  # 其实是0 1 2 三个类。等价 第一行 001 第二行 010
# 使用sparse_softmax_cross_entropy_with_logits()不需要将真实标签转为one_hot
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5=", sess.run(result5), "\n")

# 重点: 以上的交叉熵并没有计算完, 因为此时他们还是向量.我们需要将其累加成标量进行反向传播
# 注意！！！这个函数的返回值并不是一个数，而是一个向量，
# 如果要求交叉熵loss，我们要对向量求均值，
# 就是对向量再做一步tf.reduce_mean操作
loss = tf.reduce_mean(result1)
with tf.Session() as sess:
    print("loss=", sess.run(loss))

labels = [[0, 0, 1], [0, 1, 0]]
loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled), 1))
with tf.Session() as sess:
    print("loss2=", sess.run(loss2))
