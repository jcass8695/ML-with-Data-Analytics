#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import os

import tempfile
import config as cf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from statistics import mean

# Disable TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SPLIT_METHOD = cf.FLAGS['TEN_FOLD_CROSS']
# SPLIT_METHOD = cf.FLAGS['SEVENTY_THIRTY']
ALGORITHM = cf.FLAGS['LINEAR_REGRESSION']
# ALGORITHM = ''
LEARNING_RATE = 0.01
EPOCHS = 100
display_step = 50


def main():
    # Read in full dataset and convert to numpy array
    dataset = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        header=0,
        names=cf.CSV_COLUMNS_SUM,
        sep=';'
    ).values

    # Split the dataset into training and testing arrays
    # Will return lists of size 10 if using 10 fold cross validation
    x_train, x_test, y_train, y_test = split_data_frame(dataset)

    if ALGORITHM == cf.FLAGS['LINEAR_REGRESSION']:
        mse = 0
        mae = 0
        if SPLIT_METHOD == cf.FLAGS['TEN_FOLD_CROSS']:
            for i in range(10):
                mse += linear_regression_training(
                    x_train[i],
                    x_test[i],
                    y_train[i],
                    y_test[i]
                )[0]

                mae += linear_regression_training(
                    x_train[i],
                    x_test[i],
                    y_train[i],
                    y_test[i]
                )[1]

            print('Average MSE across 10 folds: {}'.format(mse / 10))
            print('Average MAE across 10 folds: {}'.format(mae / 10))

        else:
            mse, mae = linear_regression_training(
                x_train,
                x_test,
                y_train,
                y_test
            )
            print('MSE: {}'.format(mse))
            print('MAE: {}'.format(mae))


def split_data_frame(dataset):
    # Reduce number of instances to 100,000
    # And get rid of 'Instances' feature column
    dataset = np.delete(dataset, np.arange(100000, dataset.shape[0]), axis=0)
    dataset = np.delete(dataset, 0, axis=1)
    n_instances = dataset.shape[0]

    # Delete output columns for Design Matrix X
    x = np.delete(dataset, range(10, 12), axis=1)

    # Delete all feature columns for Output Matrix Y
    y = np.delete(dataset, range(10), axis=1)

    # If linear regression take the 'Target' columns otherwise take the classification labels
    if ALGORITHM == cf.FLAGS['LINEAR_REGRESSION']:
        y = np.delete(y, 1, axis=1)

    else:
        y = np.delete(y, 0, axis=1)
        y = convert_classification_labels(y)

    # First, normalize the datasets before splitting
    x = scale(x, axis=0)
    y = scale(y, axis=0)

    # Prepend the Bias feature column consisting of all 1's
    x = np.reshape(
        np.c_[np.ones(x.shape[0]), x],
        [x.shape[0], x.shape[1] + 1]
    )
    y = np.reshape(y, [x.shape[0], 1])

    if SPLIT_METHOD is cf.FLAGS['SEVENTY_THIRTY']:
        return seventy_thirty(x, y, n_instances)

    elif SPLIT_METHOD is cf.FLAGS['TEN_FOLD_CROSS']:
        return ten_fold_cross(x, y, n_instances)

    else:
        raise Exception('split_data_frame: no split_type set')


def seventy_thirty(x, y, n_instances):
    split = int(n_instances * 0.7)
    x_train = x[0:split]
    x_test = x[split:n_instances]
    y_train = y[0:split]
    y_test = y[split:n_instances]
    return x_train, x_test, y_train, y_test


def ten_fold_cross(x, y, n_instances):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    split = int(n_instances / 10)
    print(n_instances)
    for i in range(10):
        # Note there is a loss of accuracy here converting float to integer
        # The final cross validation set will not reach the final row but
        # each set size is consistent
        test_range = list(range(i * split, (i + 1) * split))
        full_range = list(range(0, n_instances))

        # training_range and test_range are the lists of indices from the whole set of instances
        # to use as the training set for this fold
        training_range = list(set(full_range) - set(test_range))
        x_train.append(x[training_range])
        x_test.append(x[test_range])
        y_train.append(y[training_range])
        y_test.append(y[test_range])

    return x_train, x_test, y_train, y_test


def convert_classification_labels(y_data):
    vfunc = np.vectorize(lambda x:
                         {
                             'Very Small Number': 1,
                             'Small Number': 2,
                             'Medium Number': 3,
                             'Large Number': 4,
                             'Very Large Number': 5
                         }[x])
    return vfunc(y_data)


def linear_regression_training(x_train, x_test, y_train, y_test):
    # TF Graph inputs
    # Take each row (instance) in x dataset
    X = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='X')

    # Take it's corresponding y output value
    Y = tf.placeholder(tf.float32, shape=[None, y_train.shape[1]], name='Y')

    # Create a column vector (no.features x 1) of weights
    W = tf.Variable(tf.ones([x_train.shape[1], 1], name='W'))

    # For each instance calculate the predicted value
    pred = tf.matmul(X, W)

    # At each prediction calculate the mean squared error
    cost = tf.reduce_mean(tf.square(pred - Y))

    # Optimize the weights after each prediction so as to minimize the MSE
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    init = tf.global_variables_initializer()

    # Start Training
    with tf.Session() as sess:
        sess.run(init)
        run_optimization(sess, optimizer, x_train, y_train, X, Y)
        mse, mae = evaluate_accuracy(sess, pred, x_test, y_test, X)

        sess.close()
        return mse, mae


def run_optimization(sess, optimizer, x_train, y_train, X, Y):
    for _ in range(EPOCHS):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})


def evaluate_accuracy(sess, pred, x_test, y_test, X):
    y_pred = sess.run(pred, feed_dict={X: x_test})
    mse = tf.reduce_mean(tf.square(y_pred - y_test))
    mae = tf.reduce_mean(tf.abs(y_pred - y_test))
    return sess.run(mse), sess.run(mae)


if __name__ == "__main__":
    main()
