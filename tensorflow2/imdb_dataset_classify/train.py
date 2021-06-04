"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
import tensorflow as tf
from tensorflow import keras
from model import MyModel
from config import set_args


@tf.function   # 加入这行认为是用图的方式执行 不加 则为eager模式  这种加了的效率比较高
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = bce_loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)   # 得到梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))   # 更新参数
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = bce_loss_func(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == '__main__':
    args = set_args()
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # Training entries: 25000, labels: 25000

    # print(train_data[0])   # [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 44...]

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,   # 用零填充
                                                            padding='post',   # 在后面填充
                                                            maxlen=256)   # 最大长度为256

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=256)

    # 使用 tf.data 来将数据集切分为 batch 以及混淆数据集
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    model = MyModel(vocab_size=10000)

    bce_loss_func = tf.keras.losses.BinaryCrossentropy()   # 损失函数
    optimizer = tf.keras.optimizers.Adam()    # 优化器

    # 衡量指标来度量模型的损失值（loss）和准确率（accuracy）
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    max_acc = 0
    for epoch in range(args.epoch_num):
        # 开始训练
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)
            print('epoch:{}, step:{}, loss:{:10f}, accuracy:{:10f}'.format(epoch, step, train_loss.result(), train_accuracy.result()))

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print('Epoch:{}, loss:{:10f}, Accuracy:{:10f}, Test_Loss:{:10f}, Test Accuracy:{:10f}'.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            test_loss.result(),
            test_accuracy.result() * 100
        ))

        if test_accuracy.result() > max_acc:
            # 保存模型
            max_acc = test_accuracy.result()
            model.save_weights('model_weights')













