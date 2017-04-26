import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib.keras import datasets
from tensorflow.contrib.slim.nets import alexnet
from tensorflow.contrib.slim.nets import resnet_v2
from tqdm import trange

# Settings
SUMMARY_DIRECTORY = ".tflogdfa/"
BATCH_SIZE = 200
slim = tf.contrib.slim

def fcnet(depth, width):
    def network(inputs, num_classes, scope='fcnet'):
        with tf.variable_scope(scope):
            net = tf.contrib.layers.flatten(inputs)

            end_points = {}
            for l in range(depth - 1):
                net = slim.fully_connected(net, width, scope="fc/fc_{}".format(l))

            net = slim.fully_connected(net, num_classes, activation_fn=None, scope="fc/fc_{}".format(depth))
            end_points[depth] = [net]

            return net, end_points

    return network

def lenet(images, num_classes=10, is_training=False, dropout_keep_prob=0.5, scope='LeNet'):
  end_points = {}
  with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout3')
    output = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')

  return output, end_points

def classification_loss(num_classes, network_outputs, targets_placeholder):
    with tf.name_scope('loss'):
        targets = tf.squeeze(tf.one_hot(targets_placeholder, num_classes), 1)
        network_outputs = tf.contrib.layers.flatten(network_outputs)
        loss = tf.reduce_mean(-targets * tf.log(network_outputs + 0.0001) - (1 - targets) * tf.log(1 - network_outputs + 0.0001))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_outputs, 1), tf.argmax(targets, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)

    return loss, [loss, accuracy], targets

def regression_loss(num_classes, network_outputs, targets_placeholder):
    with tf.name_scope('loss'):
        tf.placeholder(tf.float32, [None, 1], name='targets')
        loss = tf.reduce_mean((network_outputs - targets_placeholder)**2)
        tf.summary.scalar('loss', loss)

    return loss, [loss], targets_placeholder

experiments = [
    (datasets.cifar10, alexnet.alexnet_v2, classification_loss, 10, 'alexnet-cifar10'),
    (datasets.cifar100, alexnet.alexnet_v2, classification_loss, 100, 'alexnet-cifar100'),

    (datasets.mnist, resnet_v2.resnet_v2_200, classification_loss, 10, 'resnet200v2-mnist'),
    (datasets.cifar10, resnet_v2.resnet_v2_200, classification_loss, 10, 'resnet200v2-cifar10'),
    (datasets.cifar100, resnet_v2.resnet_v2_200, classification_loss, 100, 'resnet200v2-cifar100'),

    (datasets.mnist, lenet, classification_loss, 10, 'lenet-mnist'),
    (datasets.cifar10, lenet, classification_loss, 10, 'lenet-cifar10'),
    (datasets.cifar100, lenet, classification_loss, 100, 'lenet-cifar100'),

    (datasets.mnist, fcnet(3, 800), classification_loss, 10, 'fcnet-3x800-mnist'),
    (datasets.cifar10, fcnet(3, 800), classification_loss, 10, 'fcnet-3x800-cifar10'),
    (datasets.cifar100, fcnet(3, 800), classification_loss, 100, 'fcnet-3x800-cifar100'),
    (datasets.boston_housing, fcnet(3, 800), regression_loss, 1, 'fcnet-3x800-bostonhousing'),

    (datasets.mnist, fcnet(50, 100), classification_loss, 10, 'fcnet-50x100-mnist'),
    (datasets.cifar10, fcnet(50, 100), classification_loss, 10, 'fcnet-50x100-cifar10'),
    (datasets.cifar100, fcnet(50, 100), classification_loss, 100, 'fcnet-50x100-cifar100'),
    (datasets.boston_housing, fcnet(50, 100), regression_loss, 1, 'fcnet-50x100-bostonhousing'),
]

class Algorithms:
    @staticmethod
    def get_gdfa_tensors(exclude):
        def test(op):
            return 'Sigmoid' in op.name and 'SigmoidGrad' not in op.name

        return [op.outputs[0] for op in tf.get_default_graph().get_operations() if test(op) and op.outputs[0] not in exclude]

    @staticmethod
    def gdfa(network_output, endpoints, targets, loss_value, error_preprocessor=tf.identity):

        assert len(network_output.op.inputs) == 1, 'network output is more complicated than expected'
        assert 'BiasAdd' in network_output.op.inputs[0].name, 'final activation function might not be correctly handled'

        a_n = network_output.op.inputs[0]
        e = error_preprocessor(tf.gradients(loss_value, [a_n])[0])

        dfaloss = loss_value
        connections = Algorithms.get_gdfa_tensors(exclude=[network_output])
        print 'Total DFA Connections: {}'.format(len(connections))
        for h in connections:
            B = tf.Variable(tf.truncated_normal([h.get_shape().as_list()[1], e.get_shape().as_list()[1]], dtype=e.dtype, stddev=0.5), name='B', trainable=False)
            dfaloss += tf.reduce_mean(tf.reduce_sum(tf.matmul(B, e, transpose_b=True) * tf.transpose(h), axis=0), axis=0)

        return dfaloss

    @staticmethod
    def dfa(network_output, endpoints, targets, loss_value):
        # Note: To get literal DFA, you also need to ensure that all internal activation functions include stop_gradient
        return Algorithms.gdfa(network_output=network_output, endpoints=endpoints, targets=targets, loss_value=loss_value, error_preprocessor=tf.stop_gradient)

    @staticmethod
    def traditional(network_output, endpoints, targets, loss_value):
        return loss_value

    @staticmethod
    def lfas(network_output, endpoints, targets, loss_value):
        lfasloss = loss_value
        connections = Algorithms.get_gdfa_tensors(exclude=[network_output])
        print 'Total DFA Connections: {}'.format(len(connections))
        for h in connections:
            B = tf.Variable(tf.truncated_normal([h.get_shape().as_list()[1], targets.get_shape().as_list()[1]], dtype=h.dtype, stddev=0.5), name='B', trainable=False)
            lfasloss += tf.reduce_mean(tf.reduce_sum(tf.matmul(B, targets, transpose_b=True) * tf.transpose(h), axis=0), axis=0)

        return lfasloss

for m in range(10):
    for activation in [tf.sigmoid, tf.nn.relu]:
        for data, network, loss, output_size, experiment_name in experiments:
            for algorithm, algorithm_loss in {'gdfa': Algorithms.gdfa, 'lfas': Algorithms.lfas,
                                              'traditional': Algorithms.traditional, 'dfa': Algorithms.dfa}.items():
                name = "{}-{}-{}:{}".format(algorithm, experiment_name, activation(1.0).name[:-2].lower(), m)
                print "\n{}:".format(name)

                tf.reset_default_graph()

                (x_train, y_train), (x_test, y_test) = data.load_data()

                if len(x_train.shape) == 3:
                    x_train = np.expand_dims(x_train, 3)
                    x_test = np.expand_dims(x_test, 3)

                if len(y_train.shape) == 1:
                    y_train = np.expand_dims(y_train, 1)
                    y_test = np.expand_dims(y_test, 1)

                if y_train.dtype == np.float64:
                    y_train = y_train.astype(np.float32)
                    y_test = y_test.astype(np.float32)

                x_test_batches = np.array_split(x_test, x_test.shape[0] // BATCH_SIZE + 1)
                y_test_batches = np.array_split(y_test, x_test.shape[0] // BATCH_SIZE + 1)

                inputs_placeholder = tf.placeholder(x_train.dtype, [None] + list(x_train.shape[1:]), name='inputs')
                targets_placeholder = tf.placeholder(y_train.dtype, [None, 1], name='targets')

                if algorithm == 'dfa':
                    arg_scope = slim.arg_scope([slim.fully_connected], activation_fn=lambda x: tf.stop_gradient(activation(x)))
                else:
                    arg_scope = slim.arg_scope([slim.fully_connected], activation_fn=activation)

                with arg_scope:
                    output, endpoints = network(tf.cast(inputs_placeholder, tf.float32), num_classes=output_size)
                    output = tf.sigmoid(output)

                    loss_value, statistics, targets = loss(num_classes=output_size, network_outputs=output, targets_placeholder=targets_placeholder)

                loss_value = algorithm_loss(network_output=output, endpoints=endpoints, targets=targets, loss_value=loss_value)

                learning_rate = 1e-3
                global_step = tf.Variable(1, trainable=False)
                train = tf.train.AdamOptimizer(learning_rate).minimize(loss_value, global_step=global_step)

                sess = tf.Session()
                train_writer = tf.summary.FileWriter('{}/train/{}'.format(SUMMARY_DIRECTORY, name), sess.graph)
                test_writer = tf.summary.FileWriter('{}/test/{}'.format(SUMMARY_DIRECTORY, name))
                summary_ops = tf.summary.merge_all()
                initializer = tf.global_variables_initializer()
                sess.run(initializer)

                t = trange(10)
                for i in t:
                    batch_indicies = [np.random.randint(0, len(x_train)) for _ in range(BATCH_SIZE)]
                    x_train_batch, y_train_batch = x_train[batch_indicies], y_train[batch_indicies]

                    step_outputs = sess.run([summary_ops, loss_value, train] + statistics, feed_dict={inputs_placeholder: x_train_batch, targets_placeholder: y_train_batch})
                    train_loss = step_outputs[1]
                    train_writer.add_summary(step_outputs[0], i)

                    if (i + 1) % 50 == 0:
                        test_statistics = []
                        for x_test_batch, y_test_batch in zip(x_test_batches, y_test_batches):
                            test_outputs = sess.run([summary_ops, loss_value] + statistics, feed_dict={inputs_placeholder: x_test_batch, targets_placeholder: y_test_batch})
                            test_statistics.append(test_outputs[1:])
                        test_statistics = np.mean(test_statistics, axis=0)
                        test_loss = test_statistics[0]
                        t.set_description((("{:.5f}, " * (len(statistics) + 1))[:-2]).format(*([train_loss] + list(test_statistics[1:]))))

        # Implement early stopping?

# # Manual Improved Direct Feedback Alignment
# # with tf.name_scope('dfa'):
# #     e = tf.gradients(loss, [net[-1][1]])[0]
# #     updates = []
# #     for i, (o, net_value, x, W, b, B) in enumerate(net[:-1]):
# #         if i == len(net) - 1:
# #             B = np.eye(B.get_shape()[0]).astype(np.float64)
# #             o = net_value
# #
# #         with tf.name_scope('layer'):
# #             surogateloss = tf.reduce_sum(tf.matmul(B, tf.stop_gradient(e), transpose_b=True) * tf.transpose(o))
# #             dW, db = tf.gradients(surogateloss, [W, b])
# #             updates += [W.assign_add(learning_rate * -dW),
# #                         b.assign_add(learning_rate * -db)]
# #
# #     o, a, x, Wl, b, B = net[-1]
# #     dW, db = tf.gradients(loss, [Wl, b])
# #     updates += [Wl.assign_add(learning_rate * -dW),
# #                 b.assign_add(learning_rate * -db),
# #                 global_step.assign_add(1)]
# #
# #     dfa_train_step = tf.group(*updates)
#
# # Manual Direct Feedback Alignment
# # with tf.name_scope('dfa'):
# #     with tf.name_scope('error'):
# #         e = tf.gradients(loss, [net[-1][1]])[0]
# #
# #     updates = []
# #     if len(net) > 1:
# #         for i, (o, net_value, x, W, b, B) in enumerate(net[:-1]):
# #             with tf.name_scope('layer'):
# #                 a_prime = o * (1 - o)
# #                 error = tf.matmul(e, B, transpose_b=True) * a_prime
# #                 dW = -tf.matmul(error, x, transpose_a=True)
# #                 db = -tf.reduce_sum(error, axis=0)
# #                 updates += [W.assign_add(learning_rate * dW),
# #                             b.assign_add(learning_rate * db)]
# #
# #     # Last Layer
# #     o, a, x, Wl, b, B = net[-1]
# #     dW = -tf.matmul(e, x, transpose_a=True)
# #     db = -tf.reduce_sum(e, axis=0)
# #     updates += [Wl.assign_add(learning_rate * dW),
# #                 b.assign_add(learning_rate * db),
# #                 global_step.assign_add(1)]
# #     dfa_train_step = tf.group(*updates)