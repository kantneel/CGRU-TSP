import numpy as np 
import pandas as pd
import tensorflow as tf 

from neural_gpu import *
from graph_utils import *

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

		#return layer_norm(res, nmaps, prefix + "/" + suffix)
	reset = tf.sigmoid(conv_linear(inpts, kw, kh, nin, nin, [1, 1], True, prefix + "/r", 0.1))
	gate = tf.sigmoid(conv_linear(inpts, kw, kh, nin, nin, [1, 1], True, prefix + "/g", 0.1))
	cand = tf.tanh(conv_linear(inpts * reset, kw, kh, nin, nin, [1, 1], True, prefix + "/c", 0.0))
	'''
	if cutoff == 1.2:
		reset = sigmoid_cutoff_12(conv_lin(inpts , "r"))
		gate = sigmoid_cutoff_12(conv_lin(inpts, "g"))
	elif cutoff > 10:
		reset = sigmoid_hard(conv_lin(inpts, "r"))
		gate = sigmoid_hard(conv_lin(inpts, "g"))
	else:
		reset = sigmoid_cutoff(conv_lin(inpts, "r"), cutoff)
		gate = sigmoid_cutoff(conv_lin(inpts, "g"), cutoff)

	if cutoff > 10:
		candidate = tanh_hard(conv_lin(inpts * reset, "c", 0.0))
	else:
		# candidate = tanh_cutoff(conv_lin(inpts + [reset * mem], "c", 0.0), cutoff)
		candidate = tf.tanh(conv_lin(inpts * reset, "c", 0.0))
	'''
	return gate * inpts + (1 - gate) * cand

train_data = np.loadtxt('../graph_data/5v_data_   1_   1.csv', delimiter=',').reshape((4096, 5, 5))
train_labels = np.loadtxt('../graph_data/5v_labels_   1_   1.csv', delimiter=',').reshape((4096, 5, 5))

b_size = 4
g_size = 5

batch_shape = [b_size, g_size, g_size]
x = tf.placeholder(tf.float32, batch_shape)
y = tf.placeholder(tf.float32, batch_shape)

x_image = tf.reshape(x, [b_size, g_size, g_size, 1])
start = tf.concat((x_image, tf.zeros((b_size, g_size, 5, 9))), axis=3)

img_1 = conv_gru(start, 4, 4, 10, 10, "cgru1")
img_2 = conv_gru(img_1, 3, 3, 10, 10, "cgru2")
img_3 = conv_gru(img_2, 2, 2, 10, 10, "cgru3")
img_4 = conv_gru(img_3, 1, 1, 10, 10, "cgru4")

result = tf.reshape(img_4[:, :, :, 0], [b_size, 5, 5])
results = [tf.nn.softmax(tf.reshape(result[i], [5, 5])) for i in range(b_size)]
t_results = [results[i] + tf.transpose(results[i]) for i in range(b_size)]
labels = [tf.reshape(y[i], [5,5]) for i in range(b_size)]

v_loss = np.sum([valid_loss(results[i], 1, 1) for i in range(b_size)])
c_loss = np.sum([cycle_loss(t_results[i], labels[i], 1) for i in range(b_size)])

train_step_v = tf.train.AdamOptimizer(1e-3).minimize(v_loss)
train_step_c = tf.train.AdamOptimizer(1e-5).minimize(c_loss)



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		if (b_size * i) % 4096 > (b_size * (i + 1)) % 4096:
			continue
		batch_data = train_data[(b_size * i) % 4096 : (b_size * (i + 1)) % 4096]
		batch_labels = train_labels[(b_size * i) % 4096 : (b_size * (i + 1)) % 4096]

		v, vs = sess.run([v_loss, train_step_v], feed_dict={x : batch_data, y : batch_labels})
		c, vc = sess.run([c_loss, train_step_c], feed_dict={x : batch_data, y : batch_labels})

		if i % 20 == 0:
			print(v, c)

