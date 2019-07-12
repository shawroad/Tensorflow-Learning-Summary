"""

@file  : 001-使用动态的RNN处理变长序列.py

@author: xiaolu

@time  : 2019-07-11

"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell

tf.reset_default_graph()

# 创建输入数据
X = np.random.randn(2, 4, 5)
print(X.shape)  # (2, 4, 5)
print(X)

# 第二个样本长度为1
X[1, 1:] = 0
seq_lengths = [4, 1]
print(X.shape)  # (2, 4, 5)
print(X)


# 分别创建一个lstm与GRU的cell
cell = BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = GRUCell(3)

# 如果没有initial_state, 必须指定a的type
outputs, last_states = tf.nn.dynamic_rnn(cell, X, seq_lengths, dtype=tf.float64)
gruoutpus, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutpus, grulast_states])

print("全序列:", result[0])    # 对于全序列则输出正常长度的值  result[i]取得是第i个cell的输出
print("短序列:", result[1])    # 对于短序列, 不进行运算,直接将其填充为0

# 打印两个序列最后的状态,
print("LSTM的状态:", len(sta), sta[0], '\n', sta[1])   # 初始化中设置了state_is_tuple=true, 所以lstm的形状为(状态, 输出值)

print("GRU的短序列:", gruout[0])
print("GRU的短序列:", gruout[1])
print("GRU的状态:", len(grusta), grusta[0], '\n', grusta[1])  # 因为GRU没有状态输出, 其状态就是最终的结果,因为批次为两个, 所以输出为2



