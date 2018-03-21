import numpy as np 
import tensorflow as tf 


def conv_linear(arg, kw, kh, nin, nout, stride, do_bias, prefix, bias_start):
    # Notes 
        # arg needs to be a 3D tensor -> internal state s. 
        # kw, kh, nin, nout are just kernel specs 
        # prefix is going to be a variable scope
    with tf.variable_scope(prefix):
        with tf.device("/cpu:0"):
            k = tf.get_variable("CvK", [kw, kh, nin, nout], initializer=tf.random_normal_initializer())
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
