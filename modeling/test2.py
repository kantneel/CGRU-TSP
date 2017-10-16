import numpy as np 
import tensorflow as tf 

from neural_gpu import *
from graph_utils import *
from model_utils import *

def conv_linear(arg, kd, f_num, stride, do_bias, prefix, bias_start, l_norm):
	# Notes 
		# arg needs to be a 3D tensor -> internal state s. 
		# kw, kh, nin, nout are just kernel specs 
		# prefix is going to be a variable scope
	with tf.variable_scope(prefix):
		with tf.device("/cpu:0"):
			k = tf.get_variable("CvK", [kd, kd, f_num, f_num])
			# instantiating or getting the kernel with said shape.
			bias_term = tf.get_variable("CvB", arg.get_shape().as_list(), initializer=tf.constant_initializer(bias_start))

		res = tf.nn.convolution(arg, k, strides=stride, padding="SAME")
		#bias_term = tf.reshape(bias_term, [1, 1, 1, f_num])
		retval = res + bias_term
		if l_norm:
			retval = layer_norm(retval, f_num, prefix)
		return retval

def conv_gru(inpts, kd, f_num, prefix, layer_norm=False):
	"""Convolutional GRU."""
	reset = tf.tanh(conv_linear(inpts, kd, f_num, [1, 1], True, prefix + "/r", 0.1, layer_norm))
	gate = tf.sigmoid(conv_linear(inpts, kd, f_num, [1, 1], True, prefix + "/g", 0.1, layer_norm))
	cand = tf.nn.relu(conv_linear(inpts * reset, kd, f_num, [1, 1], True, prefix + "/c", 0.0, layer_norm))
	return gate * inpts + (1 - gate) * cand
'''
data_1 = np.loadtxt('../graph_data/4v_data_1_1.csv', delimiter=',').reshape((4096, 10, 10))
data_2 = np.loadtxt('../graph_data/6v_data_1_1.csv', delimiter=',').reshape((4096, 10, 10))

label_1 = np.loadtxt('../graph_data/4v_labels_1_1.csv', delimiter=',').reshape((4096, 10, 10))
label_2 = np.loadtxt('../graph_data/6v_labels_1_1.csv', delimiter=',').reshape((4096, 10, 10))

train_data = np.zeros((8192, 10, 10))
train_labels = np.zeros((8192, 10, 10))

for i in range(4096):
	train_data[2 * i] = data_1[i]
	train_data[2 * i + 1] = data_2[i]

	train_labels[2 * i] = label_1[i]
	train_labels[2 * i + 1] = label_2[i]
'''

train_data = np.loadtxt('../graph_data/5v_data_1_1.csv', delimiter=',').reshape((50008, 10, 10))
train_labels = np.loadtxt('../graph_data/5v_labels_1_1.csv', delimiter=',').reshape((50008, 10, 10))

b_size = 30
g_size = 10
f_num = 3
num_data = 50008

v_rate = 2e-4
c_rate = 2e-3

def train_loop(b_size, g_size, f_num, num_data, v_rate, c_rate):

	batch_shape = [b_size, g_size, g_size]
	x = tf.placeholder(tf.float32, batch_shape)
	y = tf.placeholder(tf.float32, batch_shape)

	x_image = tf.reshape(x, batch_shape + [1])
	start = x_image
	if f_num != 1:
		start = tf.concat((x_image, tf.random_normal((b_size, g_size, g_size, f_num - 1))), axis=3)

	fs = [2, 7, 2, 7, 2, 7]
	cgrus = []

	cgrus.append(conv_gru(start, fs[0], f_num, "cgru0"))
	for i in range(1, len(fs)):
		cgrus.append(conv_gru(cgrus[i - 1], fs[i], f_num, "cgru%s" % (i)))

	#img_1 = conv_gru(start, fs[0], fs[0], f_num, f_num, "cgru1")
	#img_2 = conv_gru(img_1, fs[1], fs[1], f_num, f_num, "cgru2")
	#img_3 = conv_gru(img_2, fs[2], fs[2], f_num, f_num, "cgru3")
	#img_4 = conv_gru(img_3, fs[3], fs[3], f_num, f_num, "cgru4")
	#img_5 = conv_gru(img_4, fs[4], fs[4], f_num, f_num, "cgru5")
	#img_6 = conv_gru(img_5, fs[5], fs[5], f_num, f_num, "cgru6")

	result = tf.reshape(cgrus[-1][:, :, :, 0], batch_shape)
	results = [tf.nn.softmax(tf.reshape(result[i], [g_size, g_size])) for i in range(b_size)]
	p_results = [power_and_norm(results[i], g_size) for i in range(b_size)]

	t_results = [results[i] + tf.transpose(results[i]) for i in range(b_size)]
	pt_results = [p_results[i] + tf.transpose(p_results[i]) for i in range(b_size)]
	labels = [tf.reshape(y[i], [g_size, g_size]) for i in range(b_size)]

	v_loss = np.sum([valid_loss(results[i], 0.4, 1, g_size) for i in range(b_size)]) / b_size
	c_loss = np.sum([cycle_loss(t_results[i], labels[i], 1) for i in range(b_size)]) / b_size
	r_acc = np.sum([zero_one_accuracy(pt_results[i], labels[i]) for i in range(b_size)]) / b_size

	train_step_v = tf.train.AdamOptimizer(v_rate).minimize(v_loss)
	train_step_c = tf.train.AdamOptimizer(c_rate).minimize(c_loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("EXPERIMENT VARS: ************************************")
		print(b_size, g_size, f_num, num_data, v_rate, c_rate)
		print(fs)
		print("*****************************************************")
		avgs = np.zeros(3) # v, c, r

		for i in range(50000):
			if (b_size * i) % num_data > (b_size * (i + 1)) % num_data:
				continue
			batch_data = train_data[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]
			batch_labels = train_labels[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]

			res, v, vs = sess.run([results, v_loss, train_step_v], feed_dict={x : batch_data, y : batch_labels})
			r, c, vc = sess.run([r_acc, c_loss, train_step_c], feed_dict={x : batch_data, y : batch_labels})

			avgs += np.array([v, c, r])
			if i % 100 == 0:
				avgs /= 100
				print(i, avgs)
				avgs = np.zeros(3)
				if i % 200 == 0:
					v_rate = v_rate * 0.93
					c_rate = c_rate * 0.92
				if i % 1000 == 0:
					print(np.argmax(np.round(res[0]), axis=0))
					print(np.argmax(res[0], axis=0))

train_loop(b_size, g_size, f_num, num_data, v_rate, c_rate)

