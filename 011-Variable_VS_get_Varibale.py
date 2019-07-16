"""

@file  : 011-Variable_VS_get_Varibale.py

@author: xiaolu

@time  : 2019-07-12

"""
import tensorflow as tf


# # 1. tf.Varibale()  这里的x1, x2两个变量的名字可以叫同样的名字. 执行的时候, 自动就把你自己起的名字改了
# x1 = tf.Variable(initial_value=tf.truncated_normal([1, 10], stddev=0.001),
#                  dtype=tf.float32,
#                  name='var1')
# x2 = tf.Variable(initial_value=tf.truncated_normal([1, 10], stddev=0.001),
#                  dtype=tf.float32,
#                  name='var1')
#
# # 输出变量的名字
# print(x1.name)
# print(x2.name)
#
# # 输出变量的值
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(x1))
# print(sess.run(x2))
# sess.close()


# # 2. tf.get_Varibale()  x3和x4不能叫相同的名字　若相同直接会报错　get_variable是为了进行变量共享
# x3 = tf.get_variable('get_v1',
#                      shape=[1],
#                      initializer=tf.constant_initializer(0.5),
#                      dtype=tf.float32)
# x4 = tf.get_variable('get_v2',
#                      shape=[1],
#                      initializer=tf.constant_initializer(0.4),
#                      dtype=tf.float32)
#
# print(x3.name)
# print(x4.name)
# sess2 = tf.Session()
# sess2.run(tf.global_variables_initializer())
# print(sess2.run(x3))
# print(sess2.run(x4))
# sess2.close()


# #3. 在特定作用域下获取变量
# with tf.variable_scope('test1'):
#     var1 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
#
# with tf.variable_scope('test2'):
#     var2 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
#
# print("变量1名字:", var1.name)   # 变量1名字: test1/firstvar:0
# print("变量2名字:", var2.name)   # 变量2名字: test2/firstvar:0
#
# # 在上一步的基础上实现变量共享
# with tf.variable_scope('test1', reuse=True):
#     # 当重用某个变量的时候, 必须变量名, 形状, 类型　要完全相同.
#     var3 = tf.get_variable('firstvar', shape=[2], dtype=tf.float32)
# print("变量3重用的是变量１(他们指的是同一个变量):", var3.name)


# 4. 初始化共享变量
# 看看,当前在共享变量定义的时候, 我们只是指定了shape, 并没有为其初始化,
# 但是我们在作用于中进行了初始化, 那共享变量就用它进行初始化了
with tf.variable_scope('test1', initializer=tf.constant_initializer(0.4)):
    # 共享变量可以设置shape　然后进行填充
    var1 = tf.get_variable('param1', shape=[2, 10], dtype=tf.float32)
    var2 = tf.Variable(initial_value=tf.truncated_normal([1, 10], stddev=0.001), dtype=tf.float32, name='param2')

sess3 = tf.Session()
sess3.run(tf.global_variables_initializer())
print('共享变量var1的名字为:', var1.name)
print('普通变量var2的名字为:', var2.name)
print("共享变量的值为:", sess3.run(var1))
print("普通变量的值为:", sess3.run(var2))
sess3.close()

