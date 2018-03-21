import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import linalg_ops
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
        retval = res + bias_term
        if l_norm:
            retval = layer_norm(retval, f_num, prefix)
        return retval

def conv_gru(inpts, kd, f_num, prefix, layer_norm=False):
    """Convolutional GRU."""
    reset = tf.nn.tanh(conv_linear(inpts, kd, f_num, [1, 1], True, prefix + "/r", 0.1, layer_norm))
    gate = tf.sigmoid(conv_linear(inpts, kd, f_num, [1, 1], True, prefix + "/g", 0.1, layer_norm))
    cand = tf.nn.relu(conv_linear(inpts * reset, kd, f_num, [1, 1], True, prefix + "/c", 0.0, layer_norm))
    return gate * inpts + (1 - gate) * cand



b_size = 8
g_size = 15
f_num = 3
num_data = 10000

v_rate = 6e-4
c_rate = 5 * v_rate

TRAIN_DIR = '/tmp/data'

def train_loop(b_size, g_size, f_num, num_data, v_rate, c_rate):
    curr_count = 0
    train_data = np.loadtxt('../graph_data/15_%s_data.csv' % curr_count, delimiter=',').reshape((num_data, g_size, g_size))
    train_labels = np.loadtxt('../graph_data/15_%s_labels.csv' % curr_count,  delimiter=',').reshape((num_data, g_size, g_size))


    batch_shape = [b_size, g_size, g_size]
    x = tf.placeholder(tf.float32, batch_shape)
    y = tf.placeholder(tf.float32, batch_shape)

    x_image = tf.reshape(x, batch_shape + [1])
    start = x_image
    if f_num != 1:
        #start = tf.concat_v2((x_image, tf.random_normal((b_size, g_size, g_size, f_num - 1))), axis=3)
        start = tf.concat((x_image, tf.random_normal((b_size, g_size, g_size, f_num - 1))), axis=3)

    fs = [2, 2, 2, 7, 2, 2, 2, 7, 2, 2, 2, 7, 2, 2, 2, 7]
    cgrus = []

    cgrus.append(conv_gru(start, fs[0], f_num, "cgru0"))
    for i in range(1, len(fs)):
        cgrus.append(conv_gru(cgrus[i - 1], fs[i], f_num, "cgru%s" % (i)))

    result = tf.reshape(cgrus[-1][:, :, :, 0], batch_shape)
    results = [tf.nn.softmax(tf.reshape(result[i], [g_size, g_size])) for i in range(b_size)]
    p_results = [power_and_norm(results[i], g_size) for i in range(b_size)]

    t_results = [results[i] + tf.transpose(results[i]) for i in range(b_size)]
    pt_results = [p_results[i] + tf.transpose(p_results[i]) for i in range(b_size)]
    labels = [tf.reshape(y[i], [g_size, g_size]) for i in range(b_size)]
    
    with tf.device("/cpu:0"):
        v_loss = np.sum([valid_loss(results[i], 0.4, 1, g_size) for i in range(b_size)]) / b_size
        with tf.name_scope("Validity_Loss") as scope:
            tf.summary.scalar('validity_loss', v_loss)
        c_loss = np.sum([cycle_loss2(t_results[i], x_image[i], labels[i], 15, 0.3) for i in range(b_size)]) / b_size
        with tf.name_scope("Cycle_Loss") as scope:
            tf.summary.scalar('cycle_loss', c_loss)
        r_acc = np.sum([at_least_label_accuracy(pt_results[i], x_image[i], labels[i], 15) for i in range(b_size)]) / b_size
        with tf.name_scope("Rounded_Accuracy") as scope:
            tf.summary.scalar('rounded_accuracy', r_acc)

    train_step_v = tf.train.AdamOptimizer(v_rate).minimize(v_loss)
    train_step_c = tf.train.AdamOptimizer(c_rate).minimize(c_loss)
    summ = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        #with tf
        #with tf.name_scope("global") as scope:
        sess.run(tf.global_variables_initializer())
        print("EXPERIMENT VARS: ************************************")
        print(b_size, g_size, f_num, num_data, v_rate, c_rate)
        print(fs)
        print("*****************************************************")
        avgs = np.zeros(3) # v, c, r

        summary_writer = tf.summary.FileWriter(TRAIN_DIR, sess.graph)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        for i in range(500000):
            if (b_size * i) % num_data > (b_size * (i + 1)) % num_data:
                indices = np.arange(num_data)
                np.random.shuffle(indices)
                train_data = train_data[indices]
                train_labels = train_labels[indices]
                c_rate = max(c_rate * 0.7, 5e-6)
                v_rate = min(v_rate * 0.75, c_rate * 0.9) 
                print(v_rate, c_rate)
                continue
            batch_data = train_data[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]
            batch_labels = train_labels[(b_size * i) % num_data : (b_size * (i + 1)) % num_data]

            res, v, vs = sess.run([results, v_loss, train_step_v], feed_dict={x : batch_data, y : batch_labels})
            s, r, c, vc = sess.run([summ, r_acc, c_loss, train_step_c], feed_dict={x : batch_data, y : batch_labels})
            avgs += np.array([v, c, r])
            if i % 10 == 0:
                summary_writer.add_summary(s, i)
            if i % 100 == 0:
                avgs /= 100
                if avgs[2] > 0.85:
                    print("***************")
                    print("Next Lesson")
                    print("***************")
                    curr_count += 1
                    train_data = np.loadtxt('../graph_data/15_%s_data.csv' % curr_count, delimiter=',').reshape((num_data, g_size, g_size))
                    train_labels = np.loadtxt('../graph_data/15_%s_labels.csv' % curr_count, delimiter=',').reshape((num_data, g_size, g_size))
                print(i, avgs)

                avgs = np.zeros(3)
                print(np.argmax(np.round(res[0]), axis=0))
                print(np.argmax(res[0], axis=0))

train_loop(b_size, g_size, f_num, num_data, v_rate, c_rate)

