import numpy as np 
import pandas as pd
import tensorflow as tf 

from neural_gpu import *
from graph_utils import *
from model_utils import *

train_data = np.loadtxt('../sharpening/10v_data_1_1.csv', delimiter=',').reshape((40000, 100))
train_labels = np.loadtxt('../sharpening/10v_labels_1_1.csv', delimiter=',').reshape((40000, 10, 10))

b_size = 32
g_size = 10
num_data = 40000

x = tf.placeholder(tf.float32, [None, g_size ** 2])
y = tf.placeholder(tf.float32, [None, g_size, g_size])


fc_1 = tf.get_variable("fc_1", [100, 200])
b_1 = tf.get_variable("b_1", [200], initializer=tf.constant_initializer(0.1))
r_1 = tf.nn.relu(tf.matmul(x, fc_1) + b_1)

fc_2 = tf.get_variable("fc_2", [200, 200])
b_2 = tf.get_variable("b_2", [200], initializer=tf.constant_initializer(0.1))
r_2 = tf.nn.relu(tf.matmul(r_1, fc_2) + b_2)

fc_3 = tf.get_variable("fc_3", [200, 100])
b_3 = tf.get_variable("b_3", [100], initializer=tf.constant_initializer(0.1))
r_3 = tf.reshape(tf.matmul(r_2, fc_3) + b_3, [-1, 10, 10])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=r_3)
reduced = tf.reduce_sum(loss) / b_size

train_step = tf.train.AdamOptimizer(1e-3).minimize(reduced)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	avg = 0
	for i in range(100000):
		if (b_size * i) % num_data > (b_size * (i + 1)) % num_data:
			continue
		batch_data = train_data[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]
		batch_labels = train_labels[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]

		sl, l, t = sess.run([loss, reduced, train_step], feed_dict={x : batch_data, y : batch_labels})
		avg += l

		if i % 100 == 0:
			avg /= 100 
			print(i, avg)
			avg = 0
			


