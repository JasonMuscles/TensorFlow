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

# a = tf.constant(3.0, name="a")
#
# b = tf.constant(4.0, name="b")
#
# c = tf.add(a, b, name="add")
#
# var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="variable")
# print(a, var)
#
# # 必须做一个OP初始化显示
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 必须运行初始化op
#     sess.run(init_op)
#
#     # 把程序的图结构写入时间文件，graph：把指定的图写进时间文件中
#     filewriter = tf.summary.FileWriter("./summary/test/", graph=sess.graph)
#     print(sess.run([c, var]))

def myregression():
    """
    实现一个线性回归预测
    """
    # 1.准备数据，x特征值【100，10】y目标值【100】
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="X_data")

    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 2.建立线性回归模型 1个特征，1个权重，1个偏值 y = x w + b
    # 随机给一个权重和偏值的值，让他去计算损失，然后在当前状态下优化
    # 用变量定义才能优化
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))

    bias = tf.Variable(0.0, name="b")

    y_predict = tf.matmul(x, weight) + bias

    # 3.建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4.梯度下降优化损失 leaning_rate：0 ~ 1,2,3,5,7,10
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为：%f,偏置为：%f" % (weight.eval(), bias.eval()))

        # 循环训练 运行优化
        for i in range(10):
            sess.run(train_op)
            print("第%d次优化 -- 参数权重为：%f,偏置为：%f" % (i, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    myregression()


