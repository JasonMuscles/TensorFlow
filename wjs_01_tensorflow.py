import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# OP：只要使用TensorFlow的API定义的函数都是OP
# tensor：就指代的是数据
#
# g = tf.Graph()
#
# print("G：", g)
# with g.as_default():
#     c = tf.constant(11.0)
#     print("C：", c.graph)
#
# # 实现一个加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
# sum1 = tf.add(a, b)
#
# # 默认的这张图，相当于给程序分配一段内存
# graph = tf.get_default_graph()
# print(graph)
# # 创建一张图,即：一张图包含了一组OP和Tensor，(上下文管理器)
# # 只能运行一个图，可以在会话中制定图去运行
# # 只要有会话的上下文环境，就可以方便的使用eval()
#
# # 训练模型
# # 实施的提供数据进行训练
#
# # placeholder是一个占位符
# plt = tf.placeholder(tf.float32, [None, 3])  # [X，Y]=>【X：行,Y:列】
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6]]}))  # print(sum1.eval())
#     # print(sess.run([a, b, sum1]))
    # print(a.graph)
    # print(b.graph)
    # print(sum1.graph)
    # print(sess.graph)

# TensorFlow：打印出来的形状表示
# 0维：()
# 1维：(5)
# 2维：(5,6)5行，6列
# 3维:(2,3,4)2张，3行，4列的表


# 形状的概念
# 静态形状和动态形状
# 对于静态形状来说，一旦张量形状固定了，不能再次设置静态形状
# 对于动态形状可以去创建一个新的张量,注意原数数量要匹配

# plt = tf.placeholder(tf.float32, [None, 2])
#
# print(plt, "< 1")
#
# plt.set_shape([3, 2])
#
# print(plt, "< 2")
#
# # plt.set_shape([2, 2])  # 不能再次修改了
#
# plt1 = tf.reshape(plt, [2, 3, 1])
# print(plt1, "< 3")
#
# with tf.Session() as sess:
#     pass


# 变量OP
# 变量能持久化保持，普通张量op不可以
# 当定义一个变量op时，一定要在会话中去运行初始化
# name参数：在tensorboard使用的时候显示名字，可以让相同op名字的进行区分

a = tf.constant(3.0, name="a")

b = tf.constant(4.0, name="b")

c = tf.add(a, b, name="add")

var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="variable")
print(a, var)

# 必须做一个OP初始化显示
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)

    # 把程序的图结构写入时间文件，graph：把指定的图写进时间文件中
    filewriter = tf.summary.FileWriter("./summary/test/", graph=sess.graph)
    print(sess.run([c, var]))

