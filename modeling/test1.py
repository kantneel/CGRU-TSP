import numpy as np 
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

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

		#print(arg.get_shape().as_list())
		#print(k.get_shape().as_list())
		res = tf.nn.convolution(arg, k, strides=stride, padding="SAME")
		#print(res.get_shape().as_list())
		#print("\n")
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

use_cgru = True


x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

if use_cgru:
	h_conv4 = conv_gru(h_conv2, 5, 5, 64, 64, "cgru1")
	h_conv3 = conv_gru(h_conv4, 5, 5, 64, 64, "cgru2")
	

else: 
	W_conv3 = weight_variable([5, 5, 64, 64])
	b_conv3 = bias_variable([64])
	h_conv4 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

	W_conv4 = weight_variable([5, 5, 64, 64])
	b_conv4 = bias_variable([64])
	h_conv3 = tf.nn.relu(conv2d(h_conv4, W_conv4) + b_conv4)



h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1001):
		batch = mnist.train.next_batch(5)
		if i % 100 == 0:
			val_batch = mnist.validation.next_batch(500)
			val_accuracy = accuracy.eval(feed_dict={
				x: val_batch[0], y: val_batch[1], keep_prob: 1.0})
			print('step %d, validation accuracy %g' % (i, val_accuracy))
		train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

	#print('test accuracy %g' % accuracy.eval(feed_dict={
		#x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))


