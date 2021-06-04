"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
import tensorflow as tf
import pandas as pd
from config import set_args
from tensorflow import keras
from model import MyModel


def norm(x):
    # 标准化
    return (x - train_stats['mean']) / train_stats['std']


@tf.function   # 加入这行认为是用图的方式执行 不加 则为eager模式  这种加了的效率比较高
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)   # 得到梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))   # 更新参数
    train_loss(loss)


@tf.function
def test_step(features, labels):
    predictions = model(features)
    t_loss = loss_func(labels, predictions)
    test_loss(t_loss)


if __name__ == '__main__':
    args = set_args()

    # 1. 下载数据集
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    # print(dataset.tail())   # 看后五条数据

    # print(dataset.isna().sum())   # 数据集中确实的数据量
    dataset = dataset.dropna()    # 如果有确实数据集  可以通过此代码删除确实数据的行

    # 将类别数据集转为one-hot的形式 弄成多个特征
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    # print(dataset.tail())

    # 切分数据集
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # 看一下数据的统计量
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    # 将标签脱离开来
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    print(test_labels)

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # 使用 tf.data 来将数据集切分为 batch 以及混淆数据集
    train_ds = tf.data.Dataset.from_tensor_slices((normed_train_data, train_labels)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((normed_test_data, test_labels)).batch(32)

    optimizer = keras.optimizers.RMSprop(0.001)
    # loss_func = keras.losses.MSE()
    loss_func = tf.keras.losses.MSE   # 损失函数

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    model = MyModel()

    for epoch in range(args.epoch_num):
        # 开始训练
        for step, (features, labels) in enumerate(train_ds):
            train_step(features, labels)
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, train_loss.result()))

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)











