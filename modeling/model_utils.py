import tensorflow as tf 
import numpy as tf 

def valid_loss(matrix, rowcol_coef, frob_coef, v):
	frob_squared = tf.reduce_sum(tf.square(matrix))

	col_sums = tf.reduce_sum(matrix, axis=0)
	row_sums = tf.reduce_sum(matrix, axis=1)
	ones = tf.ones(v)

	rowcol_loss = rowcol_coef * (tf.nn.l2_loss(col_sums - ones) + 
								  tf.nn.l2_loss(row_sums - ones))
	frob_loss = frob_coef * tf.square(v - frob_squared)

	return rowcol_loss

def cycle_loss(res, label, cycle_coef):
	return cycle_coef * tf.nn.l2_loss(res - label)

def power_and_norm(x):
    for i in range(6):
        x = tf.pow(x, 6)
        x /= tf.reduce_sum(x, axis=1)
    return x 