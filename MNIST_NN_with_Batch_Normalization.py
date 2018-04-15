import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    #decay: use numbers closer to 1 if you have more data
    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

def build_graph(is_training):
    X = tf.placeholder(tf.float32, [None, 28 * 28]) # None: batchsize-random
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([28 * 28,256], stddev=0.01)) #distribution for init weights
    Z1 = tf.matmul(X, W1)
    BN1 = batch_norm_wrapper(Z1, is_training)
    L1 = tf.nn.relu(BN1)
    L1 = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.random_normal([256,256], stddev=0.01))
    Z2 = tf.matmul(L1, W2)
    BN2 = batch_norm_wrapper(Z2, is_training)
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.random_normal([256,10], stddev=0.01))
    model = tf.matmul(L2, W3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=model, labels=Y))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return (X, Y), optimizer, cost, accuracy, model, keep_prob, tf.train.Saver()

print('Training Start')
tf.reset_default_graph()
(X, Y), optimizer, cost, accuracy, model, keep_prob, saver = build_graph(is_training=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):
        total_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            pred, cost_val = sess.run([optimizer, cost],
                                    feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.6})
            total_cost += cost_val
        print('Epoch:', '%04d' % (epoch + 1),
            'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

    print('Training Done')
    saved_model = saver.save(sess, './2018-03-30')

print('Testing Start')
tf.reset_default_graph()
(X, Y), _, cost, accuracy, model, keep_prob, saver = build_graph(is_training=False)

predictions = []
correct = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './2018-03-30')
    print('Accuracy:', sess.run(accuracy,
                    feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0}))

#     for i in range(100):
#         pred, corr = sess.run([tf.arg_max(model,1), accuracy],
#                              feed_dict={X: [mnist.test.images[i]], Y: [mnist.test.labels[i]], keep_prob:1.0})
#         correct += corr
#         predictions.append(pred[0])

# print("Predictions:", predictions)
# print("Accuracy:", correct/100)
