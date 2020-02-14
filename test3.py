import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def myregression():
    """
    自实现线性回归
    @return:
    """
    # 1.准备数据 x特征值[100,1] y目标值[100]个
    x = tf.compat.v1.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

    # 矩阵相乘必须是二维(y为目标值)
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 2.线性回归模型（只有一个特征，故只有一个权重外加一个偏置）即： y = x*w + bias
    # 随机初始化权重W和偏置bias去计算损失
    # 用变量定义才能优化
    weight = tf.Variable(tf.compat.v1.random_normal([1, 1], mean=0.0, stddev=1.0), name="w_weight")
    bias = tf.Variable(0.0, name="b_bias")

    y_predict = tf.matmul(x, weight) + bias

    # 3.建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4.梯度下降优化损失（0~1）一般很小大部分不超过10
    train_op = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.compat.v1.global_variables_initializer()

    # 通过会话运行程序
    with tf.compat.v1.Session() as sess:
        # 必须初始化变量（打印随机最先初始化的权重和偏置）
        sess.run(init_op)
        print("随机初始化的参数权重为：%f,偏置为：%f" % (weight.eval(), bias.eval()))

        # 循环训练-运行优化
        for i in range(500):
            sess.run(train_op)
            tf.summary.FileWriter("./summary/test2/", graph=sess.graph)
            print("第%d次优化 -- 参数权重为：%f,偏置为：%f" % (i, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    myregression()
