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

# 定义命令行参数
# 1.先定义有哪些参数需要在运行时指定
# 2.程序当中获取定义命令行参数

# 第一个参数：名字、默认值、说明
# tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
# tf.app.flags.DEFINE_string("model_dir", 100, "模型文件的加载路径")
#
# FLAGS = tf.app.flags.FLAGS

def myregression():
    """
    定义一个自回归函数
    @return:
    """
    with tf.compat.v1.variable_scope("Data"):
        # 1.准备数据x为特征数据[100, 1],y目标值
        x = tf.compat.v1.random_normal([100, 1], mean=1.75, stddev=0.5, name="X_data")

        y_true = tf.compat.v1.matmul(x, [[0.7]]) + 0.8

    with tf.compat.v1.variable_scope("Model"):
        # 2.建立模型 线性回归 1个特征  1个权重  1个偏置 y_predict = x * w + b
        # 随机一个权重和一个偏置
        weight = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 1], mean=0.0, stddev=1.0), name="W_weight")

        bias = tf.compat.v1.Variable(0.0, name="B_bias")

        y_predict = tf.compat.v1.matmul(x, weight) + bias

    with tf.compat.v1.variable_scope("Lose"):
        # 3.求损失率
        loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(y_true - y_predict))

    with tf.compat.v1.variable_scope("Optimizer"):
        # 4.优化损失
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 一、收集tensor
    tf.compat.v1.summary.scalar("Losses", loss)
    tf.compat.v1.summary.histogram("Weights", weight)
    # tf.compat.v1.summary.histogram("Biases", bias)

    # 二、合并变量写入事件文件
    merged = tf.compat.v1.summary.merge_all()


    # 初始化Variable
    init_op = tf.compat.v1.global_variables_initializer()

    # 定义一个保存模型的实例
    saver = tf.compat.v1.train.Saver()


    # 会话开启优化
    with tf.compat.v1.Session() as sess:
        # 初始化
        sess.run(init_op)
        print("W = %f | B = %f" % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewriter = tf.compat.v1.summary.FileWriter("./summary/test/", graph=sess.graph)

        # 加载模型，覆盖当前自定义随机初始值，从上次终止的参数开始
        if os.path.exists("./summary/ckpt/checkpoint"):
            saver.restore(sess, "./summary/ckpt/model")
        # 循环优化
        for i in range(500):
            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)
            filewriter.add_summary(summary, i)

            print("已优化%d次 --> W = %f | B = %f" % (i, weight.eval(), bias.eval()))

        saver.save(sess, "./summary/ckpt/model")
    return None


if __name__ == '__main__':
    myregression()


