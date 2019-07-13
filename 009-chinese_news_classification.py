"""

@file  : 009-chinese_news_classification.py

@author: xiaolu

@time  : 2019-07-12

"""
import glob
import jieba
import json
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def load_data(path):
    text = []
    label = []
    with open(path, 'r', encoding='gb18030') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.replace('\n', '').split(',')[-1]
            text.append(line)
            label.append(path[-6:-4])
    return text, label


def label2id(labels):
    # 将标签转为id
    sign = list(set(labels))
    sign2id = {}
    for i, s in enumerate(sign):
        sign2id[s] = i
    print(sign2id)

    # 标签转id
    result_labels = []
    for i in labels:
        num = sign2id.get(i)
        result_labels.append(num)
    return result_labels, sign2id


def process_data(data):
    text = ' '.join(data)
    words = jieba.lcut(text)
    words = list(set(words))
    # Padding 为0 表示这个词不在此此表
    word2id = {}
    for i, w in enumerate(words):
        word2id[w] = i+1

    # print(word2id)
    # print(len(word2id))  # 17w

    # 将文本转为id序列
    data_num = []
    for d in data:
        temp = [word2id.get(voc, 0) for voc in jieba.lcut(d)]
        data_num.append(temp)
    return word2id, data_num


def build_model(x_train, y_train):
    #
    # def cells(reuse=False):
    #     return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

    # 1.占位符
    X = tf.placeholder(tf.int32, [None, None])
    Y = tf.placeholder(tf.float32, [None, 5])   # 五个类别

    # 2.词嵌入
    embedded_size = 128  # 嵌入的维度
    encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))  # 词嵌入的输入
    encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, X)

    # 3.构造多层RNN结果
    cell = []
    for i in range(2):
        # 搞个两层的rnn
        cell.append(tf.contrib.rnn.GRUCell(64))

    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cell)
    outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

    # 4. 构造全连接层
    outputs = tf.transpose(outputs, [1, 0, 2])
    pred = tf.contrib.layers.fully_connected(outputs[-1], 5, activation_fn=None)

    # 5. 定义损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

    # 6. 优化器
    learning_rate = 0.01
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # 7. 看有那些预测正确
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    # 正确率
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 启动session
    batch_size = 128
    for i in range(5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 1
            while step * batch_size < len(x_train):
                batch_x, batch_y = x_train[batch_size * (step - 1): batch_size * step], y_train[batch_size * (
                            step - 1): batch_size * step]

                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                display_step = 10   # 每隔10步显示一次
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})

                    loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
                    print(
                        'Iter:' + str(step * batch_size) + ', Minbatch_loss: ' + str(loss) + ', Training Accuracy: ' + str(
                            acc))

                step += 1
            print(" Finished!")


if __name__ == '__main__':
    process = False
    if process == True:
        path = glob.glob('./data/今日头条新闻数据/*.csv')
        data = []
        labels = []
        for p in path:
            temp, _ = load_data(p)
            data.extend(temp)
            labels.extend(_)
        # print(len(data))    # 10398
        # print(len(labels))    # 10398

        # 1. 将标签转为数字id
        labels, sign2id = label2id(labels)
        print(labels)

        # 2. 将文本分词　然后转为字表　
        word2id, data_num = process_data(data)
        json.dump([sign2id, labels, word2id, data_num], open('./data_source.json', 'w'))

    else:
        sign2id, labels, word2id, data_num = json.load(open('./data_source.json', 'r'))

    dict_size = len(word2id)

    x_train = np.array(data_num)
    x_train = pad_sequences(x_train, padding='post', maxlen=50)
    y_train = np.array(labels)
    y_train = to_categorical(y_train)

    build_model(x_train, y_train)
