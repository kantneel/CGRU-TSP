import numpy as np 
import tensorflow as tf 

from neural_gpu import *
from graph_utils import *
from model_utils import *

def conv_linear(arg, kw, kh, nin, nout, stride, do_bias, prefix, bias_start):
	# Notes 
		# arg needs to be a 3D tensor -> internal state s. 
		# kw, kh, nin, nout are just kernel specs 
		# prefix is going to be a variable scope
	with tf.variable_scope(prefix):
		with tf.device("/cpu:0"):
			k = tf.get_variable("CvK", [kw, kh, nin, nout])
			# instantiating or getting the kernel with said shape.
			bias_term = tf.get_variable("CvB", [nout], initializer=tf.constant_initializer(bias_start))

		res = tf.nn.convolution(arg, k, strides=stride, padding="SAME")
		bias_term = tf.reshape(bias_term, [1, 1, 1, nout])
		return res + bias_term

def conv_gru(inpts, kw, kh, nin, nout, prefix):
	"""Convolutional GRU."""
	reset = tf.sigmoid(conv_linear(inpts, kw, kh, nin, nin, [1, 1], True, prefix + "/r", 0.1))
	gate = tf.sigmoid(conv_linear(inpts, kw, kh, nin, nin, [1, 1], True, prefix + "/g", 0.1))
	cand = tf.tanh(conv_linear(inpts * reset, kw, kh, nin, nin, [1, 1], True, prefix + "/c", 0.0))
	return gate * inpts + (1 - gate) * cand

train_data = np.loadtxt('../graph_data/10v_data_1_1.csv', delimiter=',').reshape((4096, 10, 10))
train_labels = np.loadtxt('../graph_data/10v_labels_1_1.csv', delimiter=',').reshape((4096, 10, 10))

b_size = 10
g_size = 10
num_data = 4096

batch_shape = [b_size, g_size, g_size]
x = tf.placeholder(tf.float32, batch_shape)
y = tf.placeholder(tf.float32, batch_shape)

x_image = tf.reshape(x, batch_shape + [1])
start = tf.concat((x_image, tf.random_normal((b_size, g_size, g_size, 19))), axis=3)

img_1 = conv_gru(start, 5, 5, 20, 20, "cgru1")
img_2 = conv_gru(img_1, 5, 5, 20, 20, "cgru2")
img_3 = conv_gru(img_2, 3, 3, 20, 20, "cgru3")
img_4 = conv_gru(img_3, 2, 2, 20, 20, "cgru4")

result = tf.reshape(img_4[:, :, :, 0], batch_shape)
results = [power_and_norm(tf.reshape(result[i], [g_size, g_size]), g_size) for i in range(b_size)]
t_results = [results[i] + tf.transpose(results[i]) for i in range(b_size)]
labels = [tf.reshape(y[i], [g_size, g_size]) for i in range(b_size)]

v_loss = np.sum([valid_loss(results[i], 0, 1, g_size) for i in range(b_size)]) / b_size
c_loss = np.sum([cycle_loss(t_results[i], labels[i], 1) for i in range(b_size)]) / b_size

train_step_v = tf.train.AdamOptimizer(1e-2).minimize(v_loss)
train_step_c = tf.train.AdamOptimizer(1e-7).minimize(c_loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	avg = 0
	for i in range(10000):
		if (b_size * i) % num_data > (b_size * (i + 1)) % num_data:
			continue
		batch_data = train_data[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]
		batch_labels = train_labels[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]

		res, v, vs = sess.run([results, v_loss, train_step_v], feed_dict={x : batch_data, y : batch_labels})
		c, vc = sess.run([c_loss, train_step_c], feed_dict={x : batch_data, y : batch_labels})

		avg += v
		if i % 20 == 0:
			avg /= 20
			print(i, avg, c)
			avg = 0
			if i % 200 == 0:
				print(np.round(2 * res[0]))

