"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input


class MyModel(Model):
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.emb = Embedding(vocab_size, 16)   # 词嵌入维度16
        self.pooling = GlobalAveragePooling1D()
        self.d1 = Dense(16, activation='relu')
        self.d2 = Dense(1, activation='sigmoid')

    def call(self, x,  **kwargs):
        x = self.emb(x)
        x = self.pooling(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
