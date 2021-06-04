"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(1)

    def call(self, inputs, **kwargs):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.d3(x)
