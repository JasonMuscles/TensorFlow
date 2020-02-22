import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def myregression():
    """
    自检线性回归
    @return:
    """
    with tf.compat.v1.variable_scope("Date"):
        # 1.准备数据。特征值：x[100，1], 目标值y [100]
        x = tf.compat.v1.random_normal([100, 1], mean=1.75, stddev=0.5, name="X_Data")
        y_true = tf.compat.v1.matmul(x, [[0.7]]) + 0.8

    with tf.compat.v1.variable_scope("Model"):
        # 2.建立模型 线性回归 1个特征 1个权重  1个偏置 y_predict = x * w + b
        # 随机一个权重和一个偏置
        weight = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 1], mean=0.0, stddev=1.0), name="W_weight")

        bias = tf.compat.v1.Variable(0.0, name="B_bias")

        y_predict = tf.compat.v1.matmul(x, weight) + bias

    with tf.compat.v1.variable_scope("Loss"):
        # 3.计算损失
        loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(y_true - y_predict))

    with tf.compat.v1.variable_scope("Train"):
        # 4.梯度优化损失
        train_op = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)

        init_op = tf.compat.v1.global_variables_initializer()

    # 收集tensor
    tf.compat.v1.summary.scalar("LOSS", loss)
    tf.compat.v1.summary.histogram("WEIGHT", weight)
    # 合并
    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:
        # 初始化
        sess.run(init_op)
        print("初始值：W=%f B=%f" % (weight.eval(), bias.eval()))

        # 建立事件
        writefile = tf.compat.v1.summary.FileWriter("./summary/myself/", graph=sess.graph)

        # 执行优化
        for i in range(1000000):
            sess.run(train_op)
            summary = sess.run(merged)
            writefile.add_summary(summary, i)
            print("优化次数：%d：W=%f B=%f" % (i, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    myregression()
