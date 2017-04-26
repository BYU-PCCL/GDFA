import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange
tf.set_random_seed(4242)

# Settings
SUMMARY_DIRECTORY = ".tflogdfa/"
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
RUN_ID = 'relaxed'  # len(os.walk(SUMMARY_DIRECTORY + '/train').next()[1])
OUTPUT_SIZE = 10
INPUT_SIZE = 784

x_placeholder = tf.placeholder(tf.float64, [None, INPUT_SIZE], name='x')
targets_placeholder = tf.placeholder(tf.float64, [None, OUTPUT_SIZE], name='targets')
global_step = tf.Variable(1, trainable=False)
learning_rate = tf.cast(tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, .95, staircase=False), dtype=tf.float64)

def layer(x, size, activation=tf.nn.sigmoid):
    with tf.name_scope("layer"):
        shape = x.get_shape().as_list()
        W = tf.Variable(tf.zeros([size, shape[1]], dtype=tf.float64), name='W')
        B = tf.Variable(tf.truncated_normal([size, OUTPUT_SIZE], dtype=tf.float64, stddev=0.5), name='B', trainable=False)
        b = tf.Variable(tf.zeros([size], dtype=tf.float64), name='b')
        a = tf.matmul( tf.stop_gradient(x), W, transpose_b=True) + b
        return activation(a), a, x, W, b, B


net = [layer(x_placeholder, 100)]
net += [layer(net[-1][0], 100)]
net += [layer(net[-1][0], OUTPUT_SIZE)]
output = net[-1][0]

with tf.name_scope('loss'):
    loss = tf.reduce_sum(-targets_placeholder * tf.log(output + 0.0001) - (1 - targets_placeholder) * tf.log(1 - output + 0.0001))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(targets_placeholder, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)

# Automatic Improved ADDFA
a_n = net[-1][1]
e = tf.stop_gradient( tf.gradients(loss, [a_n])[0] )

dfaloss = loss
for i, (h, net_value, x, W, b, B) in enumerate(net[:-1]):
    dfaloss += tf.reduce_mean(tf.reduce_sum(tf.matmul(B, e, transpose_b=True) * tf.transpose(h), axis=0), axis=0)

dfa_train_step = tf.train.AdamOptimizer(learning_rate).minimize(dfaloss, global_step=global_step)

# Manual Improved Direct Feedback Alignment
# with tf.name_scope('dfa'):
#     e = tf.gradients(loss, [net[-1][1]])[0]
#     updates = []
#     for i, (o, net_value, x, W, b, B) in enumerate(net[:-1]):
#         if i == len(net) - 1:
#             B = np.eye(B.get_shape()[0]).astype(np.float64)
#             o = net_value
#
#         with tf.name_scope('layer'):
#             surogateloss = tf.reduce_sum(tf.matmul(B, tf.stop_gradient(e), transpose_b=True) * tf.transpose(o))
#             dW, db = tf.gradients(surogateloss, [W, b])
#             updates += [W.assign_add(learning_rate * -dW),
#                         b.assign_add(learning_rate * -db)]
#
#     o, a, x, Wl, b, B = net[-1]
#     dW, db = tf.gradients(loss, [Wl, b])
#     updates += [Wl.assign_add(learning_rate * -dW),
#                 b.assign_add(learning_rate * -db),
#                 global_step.assign_add(1)]
#
#     dfa_train_step = tf.group(*updates)

# Manual Direct Feedback Alignment
# with tf.name_scope('dfa'):
#     with tf.name_scope('error'):
#         e = tf.gradients(loss, [net[-1][1]])[0]
#
#     updates = []
#     if len(net) > 1:
#         for i, (o, net_value, x, W, b, B) in enumerate(net[:-1]):
#             with tf.name_scope('layer'):
#                 a_prime = o * (1 - o)
#                 error = tf.matmul(e, B, transpose_b=True) * a_prime
#                 dW = -tf.matmul(error, x, transpose_a=True)
#                 db = -tf.reduce_sum(error, axis=0)
#                 updates += [W.assign_add(learning_rate * dW),
#                             b.assign_add(learning_rate * db)]
#
#     # Last Layer
#     o, a, x, Wl, b, B = net[-1]
#     dW = -tf.matmul(e, x, transpose_a=True)
#     db = -tf.reduce_sum(e, axis=0)
#     updates += [Wl.assign_add(learning_rate * dW),
#                 b.assign_add(learning_rate * db),
#                 global_step.assign_add(1)]
#     dfa_train_step = tf.group(*updates)

# sgd_train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# adam_train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
# rprop_train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Initialize session and writers
sess = tf.Session()
train_writer = tf.summary.FileWriter(SUMMARY_DIRECTORY + '/train/{}'.format(RUN_ID), sess.graph)
test_writer = tf.summary.FileWriter(SUMMARY_DIRECTORY + '/test/{}'.format(RUN_ID))
summary_ops = tf.summary.merge_all()
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Data and main loop
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

t = trange(10000)
for i in t:
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    train_summary, _, l, ta, lr, tmpe = sess.run([summary_ops, dfa_train_step, loss, accuracy, learning_rate, e], feed_dict={x_placeholder: batch_xs, targets_placeholder: batch_ys})
    train_writer.add_summary(train_summary, i)

    if i % 10 == 0:
        test_summary, l, a = sess.run([summary_ops, loss, accuracy], feed_dict={x_placeholder: mnist.test.images, targets_placeholder: mnist.test.labels})
        test_writer.add_summary(test_summary, i)
        t.set_description("loss: {:5f}, test acc:{:5f}, lr:{:5f}, train acc:{:5f}".format(l, a, lr, ta))


# # ### VARIANCE EXPERIMENT ###
# tmprelaxedgrads = np.array(tmprelaxedgrads)
# tmptightgrads = np.array(tmptightgrads)
# variancesrelaxed = []
# variancestight = []
# for i in range(10, len(tmprelaxedgrads)):
#
#     variancesrelaxed.append(tmprelaxedgrads[i - 10: i].var(axis=0).mean())
#     variancestight.append(tmptightgrads[i - 10: i].var(axis=0).mean())
#
# import matplotlib.pyplot as plt
# plt.plot(variancesrelaxed, label='stop_gradient = False')
# plt.plot(variancestight, label='stop_gradient = True')
# plt.title('Variance of Gradient w.r.t parameters')
# plt.ylabel('Variance')
# plt.xlabel('Train Step')
# plt.legend()
# plt.show()
#
