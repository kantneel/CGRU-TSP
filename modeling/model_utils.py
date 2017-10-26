import tensorflow as tf 
import numpy as np 

def valid_loss(matrix, rowcol_coef, equal_coef, v):

	col_sums = tf.reduce_sum(matrix, axis=0)
	row_sums = tf.reduce_sum(matrix, axis=1)
	ones = tf.ones(v)

	subd = tf.abs(matrix - 1 / v)

	rowcol_loss = rowcol_coef * (tf.nn.l2_loss(row_sums - ones) + tf.nn.l2_loss(col_sums - ones))
	equal_loss = - equal_coef * tf.nn.l2_loss(tf.square(tf.square(subd)))

	return rowcol_loss + equal_loss

def cycle_loss(res, label, cycle_coef):
	return cycle_coef * tf.nn.l2_loss(res - label)

def cycle_loss2(res, graph, label, vertices, cycle_coef):
	good_coef = tf.cond(cycle_length_diff(res, vertices, graph, label, vertices)[0] > 0,
	 					lambda: 1, lambda: -1)
	return good_coef * cycle_coef * tf.nn.l2_loss(graph * (res-label))

def zero_one_accuracy(res, label):
	#retval = tf.select(tf.nn.l2_loss(res-label) < tf.constant(0.1, dtype=tf.float32), tf.constant(1.0), tf.constant(0.0))
	retval = tf.cond(tf.nn.l2_loss(res-label) < tf.constant(0.1, dtype=tf.float32), lambda: 1, lambda: 0)
	return retval

def at_least_label_accuracy(res, graph, label, vertices):
	rets = cycle_length_diff(res, graph, label, vertices)

	return 1 if rets[0] < 0.03 * rets[2] else 0

def cycle_length_diff(res, graph, label, vertices):
	graph_compare = tf.slice(graph, [0, 0], [vertices, vertices])
	res_compare = tf.slice(res, [0, 0], [vertices, vertices])
	label_compare = tf.slice(label, [0, 0], [vertices, vertices])

	res_length = tf.reduce_sum(graph_compare * res_compare)
	label_length = tf.reduce_sum(graph_compare * label_compare)

	out_of_bounds_fail = vertices - tf.reduce_sum(res_compare)
	exactness_fail = 1 - tf.reduce_sum(tf.square(res_compare)) / vertices
	retval = tf.cond(out_of_bounds_fail > 0.5 or exactness_fail > 0.25, 
					 lambda:tf.square(res_length - label_length), lambda:res_length - label_length)

	return retval, res_length, label_length

def power_and_norm(x, v):
	x_r = x + tf.zeros([v, v])
	for i in range(6):
		x_r = tf.pow(x_r, 2)
		x_r /= tf.reduce_sum(x_r, axis=0)

	return x_r
	#return 0.5 * x_r + 0.5 * tf.transpose(x_c)

def layer_norm(x, nmaps, prefix, epsilon=1e-5):
	"""Layer normalize the 4D tensor x, averaging over the last dimension."""
	with tf.variable_scope(prefix):
		scale = tf.get_variable("layer_norm_scale", [nmaps],
														initializer=tf.ones_initializer())
		bias = tf.get_variable("layer_norm_bias", [nmaps],
													 initializer=tf.zeros_initializer())
		mean, variance = tf.nn.moments(x, [3], keep_dims=True)
		norm_x = (x - mean) / tf.sqrt(variance + epsilon)
		return norm_x * scale + bias
