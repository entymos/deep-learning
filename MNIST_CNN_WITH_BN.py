import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def build_graph(is_training):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    L1 = tf.contrib.layers.batch_norm(L1, is_training) 
    # https://www.facebook.com/groups/TensorFlowKR/permalink/517258461948550/
    # tf.layers.batch_normlaization --> update_ops를 해주어야함,이렇게 추가할경우,
    # 학습속도가 느려지고, accuracy 올라가는 폭이 매우 좁음
    # 추가 안할경우, batch norm을 안하게 되는 경우와 같음
    # tf.contrib.layers.batch_norm --> (update_ops 옵션이 안에 따로 있어서 None을 주면 됨)
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
    L2 = tf.contrib.layers.batch_norm(L2, is_training)
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
    L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
    L3 = tf.matmul(L3, W3)
    L3 = tf.contrib.layers.batch_norm(L3, is_training)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

    W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
    model = tf.matmul(L3, W4)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) #.9908
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost) #.9919 #0.9924 #0.9932
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return (X, Y), optimizer, cost, accuracy, model, keep_prob, tf.train.Saver()

print('Load Dataset')
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

print('Training Start')
(X, Y), optimizer, cost, accuracy, model, keep_prob, saver = build_graph(is_training=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)

            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.7})
            total_cost += cost_val
        
        print('Epoch: ', '%04d' % (epoch + 1),
        'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

    print('Train Done')
    saved_model = saver.save(sess, './mnist_v3_BN_2018-04-02')

print('Testing Start')
tf.reset_default_graph()
(X, Y), _, cost, accuracy, model, keep_prob, saver = build_graph(is_training=False)

predictions = []
correct = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './mnist_v3_BN_2018-04-02')
    print('Accuracy:', sess.run(accuracy,
                    feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob:1.0}))
