import numpy as np 
import pandas as pd
import tensorflow as tf 

from neural_gpu import *
from graph_utils import *

train_data = np.loadtxt('../sharpening/10v_data_1_1.csv', delimiter=',').reshape((4096, 100))
train_labels = np.loadtxt('../sharpening/10v_labels_1_1.csv', delimiter=',').reshape((4096, 100))

b_size = 10
g_size = 10
num_data = 4096

x = tf.placeholder(tf.float32, [None, g_size ** 2])
y = tf.placeholder(tf.float32, [None, g_size ** 2])


fc_1 = tf.get_variable("fc_1", [100, 100])
b_1 = tf.get_variable("b_1", [100], initializer=tf.constant_initializer(0.1))
r_1 = tf.nn.tanh(tf.matmul(x, fc_1) + b_1)

fc_2 = tf.get_variable("fc_2", [100, 100])
b_2 = tf.get_variable("b_2", [100], initializer=tf.constant_initializer(0.1))
r_2 = tf.matmul(r_1, fc_2) + b_2

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=r_2)
reduced = tf.reduce_sum(loss)

train_step = tf.train.AdamOptimizer(1e-3).minimize(reduced)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100000):
		if (b_size * i) % num_data > (b_size * (i + 1)) % num_data:
			continue
		batch_data = train_data[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]
		batch_labels = train_labels[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]

		l, t = sess.run([reduced, train_step], feed_dict={x : batch_data, y : batch_labels})

		if i % 20 == 0:
			print(i, l)


