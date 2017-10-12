import tensorflow as tf 
import numpy as np 

def valid_loss(matrix, rowcol_coef, equal_coef, v):

	col_sums = tf.reduce_sum(matrix, axis=0)
	row_sums = tf.reduce_sum(matrix, axis=1)
	ones = tf.ones(v)

	subd = tf.abs(matrix - 1 / v)

	rowcol_loss = rowcol_coef * (tf.nn.l2_loss(row_sums - ones) + tf.nn.l2_loss(col_sums - ones))
	equal_loss = - equal_coef * tf.norm(tf.square(tf.square(subd)))

	return rowcol_loss + equal_loss

def cycle_loss(res, label, cycle_coef):
	return cycle_coef * tf.nn.l2_loss(res - label)


def power_and_norm(x, v):
	x_r = x + tf.zeros([v, v])
	x_c = tf.transpose(x + tf.zeros([v, v]))
	for i in range(6):
		x_r = tf.pow(x_r, 2)
		x_r /= tf.reduce_sum(x_r, axis=0)
	for i in range(6):
		x_c = tf.pow(x_c, 2)
		x_c /= tf.reduce_sum(x_c, axis=0)
	return 0.5 * x_r + 0.5 * tf.transpose(x_c)