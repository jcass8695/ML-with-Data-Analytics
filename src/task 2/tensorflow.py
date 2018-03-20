#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import os

import config as cf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import scale

# Disable TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATASET = cf.SUM_WITH_NOISE
# DATASET = cf.HOUSE_PRICES
# SPLIT_METHOD = cf.SEVENTY_THIRTY
SPLIT_METHOD = cf.TEN_FOLD_CROSS
# ALGORITHM = cf.LINEAR_REG
ALGORITHM = cf.KNN

LEARNING_RATE = cf.LEARNING_RATE
EPOCHS = cf.EPOCHS


def main():
    if DATASET == cf.SUM_WITH_NOISE:
        sep = cf.SUM_CSV_SEP

    else:
        sep = cf.HOUSE_CSV_SEP

    # Read in full dataset and convert to numpy array
    dataset = pd.read_csv(
        cf.PATHS[DATASET],
        header=0,
        sep=sep
    ).values

    # Split the dataset into training and testing arrays
    # Will return lists of size 10 if using 10 fold cross validation
    x_train, x_test, y_train, y_test = split_data_frame(dataset)
    if ALGORITHM == cf.LINEAR_REG:
        mse = 0
        mae = 0
        if SPLIT_METHOD == cf.TEN_FOLD_CROSS:
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

            print('Average RMSE across 10 folds: {}'.format(mse / 10))
            print('Average MAE across 10 folds: {}'.format(mae / 10))

        else:
            mse, mae = linear_regression_training(
                x_train,
                x_test,
                y_train,
                y_test
            )

            print('RMSE: {}'.format(mse))
            print('MAE: {}'.format(mae))

    elif ALGORITHM == cf.KNN:
        accuracy = 0
        f1 = 0
        if SPLIT_METHOD == cf.TEN_FOLD_CROSS:
            for i in range(10):
                accuracy += k_nearest_neighbours(
                    x_train[i],
                    x_test[i],
                    y_train[i],
                    y_test[i]
                )[0]

                f1 += k_nearest_neighbours(
                    x_train[i],
                    x_test[i],
                    y_train[i],
                    y_test[i]
                )[1]

                print('Fold: {}'.format(i + 1))

            print('Average accuracy across 10 folds: {}'.format(accuracy * 10))
            print('Average F1 across 10 folds: {}'.format(f1 / 10))

        else:
            accuracy, f1 = k_nearest_neighbours(
                x_train,
                x_test,
                y_train,
                y_test
            )

            print('Accuracy: {}%'.format(accuracy * 100))
            print('F1: {}'.format(f1))


def split_data_frame(dataset):
    if DATASET == cf.SUM_WITH_NOISE:
        x, y, dataset = format_sum_dataset(dataset)
    else:
        x, y, dataset = format_house_dataset(dataset)

    n_instances = dataset.shape[0]

    # Normalize the x data
    x = scale(x, axis=0)
    if ALGORITHM == cf.LINEAR_REG:
        y = scale(y, axis=0)
        x, y = prepend_bias_term(x, y)

    if SPLIT_METHOD is cf.SEVENTY_THIRTY:
        return seventy_thirty(x, y, n_instances)

    else:
        return ten_fold_cross(x, y, n_instances)


def format_sum_dataset(dataset):
    # Reduce number of instances to 100,000
    dataset = np.delete(
        dataset,
        np.arange(20000, dataset.shape[0]),
        axis=0
    )

    # Get rid of 'Instances' feature column
    dataset = np.delete(dataset, 0, axis=1)

    # Delete unecessary columns from X and Y
    x = np.delete(dataset, range(10, 12), axis=1)
    y = np.delete(dataset, range(10), axis=1)

    # If linear regression take the 'Target' column otherwise take the
    # 'Target Class' classification label
    if ALGORITHM == cf.LINEAR_REG:
        y = np.delete(y, 1, axis=1)

    else:
        y = np.delete(y, 0, axis=1)
        y = convert_classification_label_for_SUM(y)

    return x, y, dataset


def format_house_dataset(dataset):
    # Remove 'id' and 'date' feature columns
    dataset = np.delete(dataset, [0, 1], axis=1)

    # Remove 'price' feature from X and everything but price from y
    x = np.delete(dataset, 0, axis=1)
    y = np.delete(dataset, range(1, 19), axis=1)
    if ALGORITHM == cf.KNN:
        y = convert_classification_label_for_housing(y)

    return x, y, dataset


def prepend_bias_term(x, y):
    # Prepend the Bias feature column consisting of all 1's
    x = np.reshape(
        np.c_[np.ones(x.shape[0]), x],
        [x.shape[0], x.shape[1] + 1]
    )
    y = np.reshape(y, [x.shape[0], 1])

    return x, y


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


def convert_classification_label_for_SUM(y_data):
    vfunc = np.vectorize(lambda x:
                         {
                             'Very Small Number': 0,
                             'Small Number': 0,
                             'Medium Number': 0,
                             'Large Number': 1,
                             'Very Large Number': 1
                         }[x])
    return vfunc(y_data)


def convert_classification_label_for_housing(y_data):
    mean_house_price = y_data.mean(axis=0)
    print(mean_house_price)
    vfunc = np.vectorize(lambda x: 1 if x >= mean_house_price else 0)
    return vfunc(y_data)

'''
> For each point in x_test, look at it's K nearest neighbours in x_train
> Record the indices of these neighbours
> Pick a class for the current point in x_test by averaging the 
  classes of the neighbours which are in y_train
> Check the predicted class against the actual class in y_test
'''
def k_nearest_neighbours(x_train, x_test, y_train, y_test):
    correct_predictions = 0
    predictions_list = []
    K = cf.K_NEIGHBOURS

    X_train = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
    X_test = tf.placeholder(tf.float32, shape=[x_test.shape[1]])
    l1_distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(X_train, X_test)), axis=1))
    _, indices_of_knn = tf.nn.top_k(l1_distance, k=K, sorted=False)

    with tf.Session() as sess:
        for index, instance in enumerate(x_test):

            # Get KNN of current test point from the test set in the training set
            nn_indices = sess.run(indices_of_knn, feed_dict={
                X_train: x_train, X_test: instance})

            # Take average class of each neighbour and round to get prediction class
            average = 0
            for i in nn_indices:
                average += y_train[i][0]

            pred_class = int(round(average / K))
            predictions_list.append(pred_class)
            if pred_class == y_test[index]:
                correct_predictions += 1

        sess.close()
    accuracy = correct_predictions / x_test.shape[0]
    f1 = evaluate_f1_score_binary_class(predictions_list, y_test)
    return accuracy, f1


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
    optimizer = tf.train.GradientDescentOptimizer(cf.LEARNING_RATE).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Train model using training sets and optimize with GDA
        run_optimization(sess, optimizer, x_train, y_train, X, Y)

        # Evaluate accuracy of model using test sets with MSE and MAE
        mse, mae = evaluate_mse_mae(sess, pred, x_test, y_test, X)
        sess.close()

    return mse, mae


def run_optimization(sess, optimizer, x_train, y_train, X, Y):
    for _ in range(cf.EPOCHS):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})


def evaluate_mse_mae(sess, pred, x_test, y_test, X):
    y_pred = sess.run(pred, feed_dict={X: x_test})
    mse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_test)))
    mae = tf.reduce_mean(tf.abs(y_pred - y_test))
    return sess.run(mse), sess.run(mae)


def evaluate_f1_score_binary_class(predictions, y_test):
    try:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred, real in zip(predictions, y_test):
            if pred == 1 and real == 1:
                true_positives += 1

            elif pred == 1 and real == 0:
                false_positives += 1

            elif pred == 0 and real == 1:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (false_negatives + true_positives)
        return (2 * precision * recall) / (precision + recall)

    except ZeroDivisionError:
        print(true_positives, false_positives, false_negatives, sep='\n')
        return

if __name__ == "__main__":
    main()
