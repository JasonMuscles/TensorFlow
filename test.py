import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(11.0)
b = tf.constant(11.5)

sum1 = tf.add(a, b)

gf = tf.get_default_graph()

with tf.Session() as sess:
    print(sess.run(sum1))
