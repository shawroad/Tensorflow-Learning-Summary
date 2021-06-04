"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
import tensorflow as tf
from model import MyModel
from config import set_args


@tf.function   # 加入这行认为是用图的方式执行 不加 则为eager模式  这种加了的效率比较高
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)   # 得到梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))   # 更新参数
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == '__main__':
    args = set_args()
    # 1. 加载数据集
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0   # 归一化
    # print(x_train.shape)    # (60000, 28, 28)
    # print(x_test.shape)    # (10000, 28, 28)
    # print(y_train.shape)   # (60000,)  没有转成one-hot的那种形式

    # 多加一个通道维度
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    # print(x_train.shape)     # (60000, 28, 28, 1)   这个跟torch不太一样 torch的维度在第二维 这个在第四维

    # 使用 tf.data 来将数据集切分为 batch 以及混淆数据集
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()   # 损失函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    # 优化器

    # 衡量指标来度量模型的损失值（loss）和准确率（accuracy）
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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





