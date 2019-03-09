#split data into train, dev, test
#get embeddings for classes and each training example. Additional features from data besides text?
#write my own P/R/F1 script
#after this one, implement the same thing with PyTorch

#in this file: try/except for pickled input vectors (created in prepare_data.py). Except clause will give a message
#stating the user should run prepare_data.py first.
#take training size as command line arg and get random training samples from total training (comes from prepare_data.py)
# to pass to NN.

import tensorflow as tf
import numpy as np
import csv


def main(train_input, dev_input, test_input, train_senses, dev_senses):
    """Build neural network architecture, feed in data and train model using dev set to evaluate cross-entropy loss.
    Run trained model on test set."""

    results_file = open('results.txt', 'w')

    # set up dimensions of model. Uncomment lines to add a second hidden layer.
    input_size = len(train_input[0])
    batch_size = 100
    hidden_size_1 = 200
    # hidden_size_2 = 25
    output_size = len(train_senses[0])

    # build MLP neural network architecture (symbolic)
    print('...building the neural network...')

    w1 = tf.Variable(tf.random_uniform((input_size, hidden_size_1), -1, 1))
    b1 = tf.Variable(tf.zeros((1, hidden_size_1)))
    w2 = tf.Variable(tf.random_uniform((hidden_size_1, output_size), -1, 1))
    #w2 = tf.Variable(tf.random_uniform((hidden_size_1, hidden_size_2), -1, 1))
    b2 = tf.Variable(tf.zeros((1, output_size)))
    #b2 = tf.Variable(tf.zeros((1, hidden_size_2)))
    # w3 = tf.Variable(tf.random_uniform((hidden_size_2, output_size), -1, 1))
    # b3 = tf.Variable(tf.zeros((1, output_size)))

    x = tf.placeholder(tf.float32, (None, input_size))

    predicted = tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(x, w1) + b1), w2) + b2)
    #predicted = tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(x, w1) + b1), w2) + b2), w3) + b3)

    print('...run forward computation...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sess.run(predicted, feed_dict={x: dev_input})

    print('...training model...')

    # define cross-entropy loss (symbolic)
    gold_y = tf.placeholder(tf.float32, (None, output_size))
    cross_entropy = - tf.reduce_sum(gold_y * tf.log(predicted + .00000001), axis=1)

    # define optimizer (symbolic)
    learning_rate = 0.00003
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # execute training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    converged = False
    iteration = 0
    prev_dev_loss_value = 0
    converge_count = 0
    while not converged:
        for i in range(int(len(train_input) / batch_size)):
            sess.run(
                train_step,
                feed_dict={
                    x: train_input[i * batch_size:(i + 1) * batch_size],
                    gold_y: train_senses[i * batch_size:(i + 1) * batch_size]})

        iteration += 1

        # calculate loss on dev set after traversing all minibatches
        dev_loss = tf.reduce_sum(cross_entropy)
        dev_loss_value = sess.run(dev_loss,
            feed_dict={x: dev_input, gold_y: dev_senses})

        if iteration % 10 == 0:
            print('iteration: ', iteration, ', loss on dev:', dev_loss_value)
            results_file.write(str(iteration) + '\t' + str(dev_loss_value) + '\n')

        # convergence check
        if dev_loss_value >= prev_dev_loss_value:
            converge_count += 1
            print(converge_count)
            if converge_count >= 15:
                converged = True
                print('...training has converged...')
        else:
            converge_count = 0

        prev_dev_loss_value = dev_loss_value

    # test trained model
    print('...testing trained model...')
    results = sess.run(predicted, feed_dict={x: test_input})
    return results


if __name__ == '__main__':

    # prepare input representations or load them from disk
    try:
        with open('train_vectors.npy', 'rb') as f1, open('dev_vectors.npy', 'rb') as f2, \
                open('test_vectors.npy', 'rb') as f3, open('train_labels.npy', 'rb') as f4, \
                open('dev_labels.npy', 'rb') as f5, open('test_labels.npy', 'rb') as f6:
            print('...retrieving input vectors from disk...')

            train_input = np.load(f1)
            dev_input = np.load(f2)
            test_input = np.load(f3)
            train_labels = np.load(f4)
            dev_labels = np.load(f5)
            test_labels = np.load(f6)

        results = main(train_input, dev_input, test_input, train_labels, dev_labels)

        # TODO output a csv that looks like the test set but with predicted labels, and have a separate scorer script

    except FileNotFoundError:
        print('Data not yet loaded. Run prepare_data.py first')
