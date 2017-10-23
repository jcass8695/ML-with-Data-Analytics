#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import os
from timeit import default_timer as timer

import tempfile
import config as cf
import pandas as pd
import tensorflow as tf

# Disable TF warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SPLIT_METHOD = cf.FLAGS['SEVENTY_THIRTY']
ALGORITHM = cf.FLAGS['LINEAR_REGRESSION']


def main():
    # Read in full dataset
    df = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        header=0,
        names=cf.CSV_COLUMNS_SUM,
        sep=';'
    )

    if ALGORITHM == cf.FLAGS['LINEAR_REGRESSION']:
        # Split the dataset into training and testing
        # Will return a list of size 10 if using 10 fold cross validation
        df_training, df_testing = split_data_frame(df, SPLIT_METHOD)

        # Get the label columns, the output vectors in Tensor Flow terminology
        # Will return a list of size 10 if using 10 fold cross validation
        train_label_regression, test_label_regression = build_regression_labels(
            df_training,
            df_testing,
            SPLIT_METHOD
        )

        # Build TF feature columns
        # Will NOT return a list, feature cols constant and independant of
        # Splitting method
        base_feature_columns = build_feature_columns()

        # Build training and testing input function
        # Will return a list of size 10 if using 10 fold cross validation
        training_input_fn, testing_input_fn = build_regression_input_functions(
            df_training,
            df_testing,
            train_label_regression,
            test_label_regression,
            SPLIT_METHOD
        )

        # Do that ML
        y_pred = evaluate_model(
            base_feature_columns,
            ALGORITHM,
            training_input_fn,
            testing_input_fn,
            SPLIT_METHOD
        )

        mean_square_error(y_pred, test_label_regression)


def split_data_frame(df, split_type):
    df_depth = int(df.shape[0])
    if split_type is cf.FLAGS['SEVENTY_THIRTY']:
        split = int(df_depth * 0.7)
        df_training = df.iloc[:split, 1:]
        df_testing = df.iloc[split:, 1:]

    elif split_type is cf.FLAGS['TEN_FOLD_CROSS']:
        df_training = []
        df_testing = []
        split = int(df_depth / 10)

        # Split training and testing set into 10 separate sets
        for i in range(10):
            # Note there is a loss of accuracy here converting float to integer
            # The final cross validation set will not reach the final row but
            # each set size is consistent
            test_range = list(range(i * split, (i + 1) * split))
            full_range = list(range(0, df_depth))
            training_range = list(set(full_range) - set(test_range))
            df_training.append(df.iloc[training_range, 1:])
            df_testing.append(df.iloc[test_range, 1:])

    else:
        raise Exception('split_data_frame: no split_type set')

    return df_training, df_testing


def build_regression_labels(df_training, df_testing, split_type):
    if split_type == cf.FLAGS['SEVENTY_THIRTY']:
        return df_training['Target'], df_testing['Target']

    elif split_type == cf.FLAGS['TEN_FOLD_CROSS']:
        training_label_list = []
        testing_label_list = []
        for value in df_training:
            training_label_list.append(value['Target'])

        for value in df_testing:
            testing_label_list.append(value['Target'])

        return training_label_list, testing_label_list

    else:
        raise Exception('build_regression_labels: no split_type set')


def build_classification_labels(df_training, df_testing):
    # Format the categorical class values into integer representations
    # Note indexing a pandas DataFrame like this returns a Series object
    train = (df_training['Target_Class']
             .apply(lambda x:
                    {
                        'Very Small Number': 1,
                        'Small Number': 2,
                        'Medium Number': 3,
                        'Large Number': 4,
                        'Very Large Number': 5
                    }[x]
                    ))

    test = (df_testing['Target_Class']
            .apply(lambda x:
                   {
                       'Very Small Number': 1,
                       'Small Number': 2,
                       'Medium Number': 3,
                       'Large Number': 4,
                       'Very Large Number': 5
                   }[x]
                   ))
    return train, test


def build_feature_columns():
    feature1 = tf.feature_column.numeric_column('Feature_1')
    feature2 = tf.feature_column.numeric_column('Feature_2')
    feature3 = tf.feature_column.numeric_column('Feature_3')
    feature4 = tf.feature_column.numeric_column('Feature_4')
    feature5 = tf.feature_column.numeric_column('Feature_5')
    feature6 = tf.feature_column.numeric_column('Feature_6')
    feature7 = tf.feature_column.numeric_column('Feature_7')
    feature8 = tf.feature_column.numeric_column('Feature_8')
    feature9 = tf.feature_column.numeric_column('Feature_9')
    feature10 = tf.feature_column.numeric_column('Feature_10')
    return [feature1, feature2, feature3, feature4, feature5,
            feature6, feature7, feature8, feature9, feature10]


def build_regression_input_functions(df_training, df_testing, training_label, testing_label, split_type):
    if split_type == cf.FLAGS['SEVENTY_THIRTY']:
        training_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=df_training,
            y=training_label,
            shuffle=False
        )

        testing_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=df_testing,
            y=testing_label,
            shuffle=False
        )

    elif split_type == cf.FLAGS['TEN_FOLD_CROSS']:
        training_input_fn = []
        testing_input_fn = []
        for index, value in enumerate(df_training):
            training_input_fn.append(tf.estimator.inputs.pandas_input_fn(
                x=value,
                y=training_label[index],
                shuffle=False
            ))

        for index, value in enumerate(df_testing):
            testing_input_fn.append(tf.estimator.inputs.pandas_input_fn(
                x=value,
                y=testing_label[index],
                shuffle=False
            ))

    else:
        raise Exception('build_regression_input_functions: no split_type set')

    return training_input_fn, testing_input_fn


def evaluate_model(feature_columns, model_type, training_input_fn, testing_input_fn, split_type):
    # TODO: implement other algorithms and add to config file
    if model_type == cf.FLAGS['LINEAR_REGRESSION']:
        model = tf.estimator.LinearRegressor(
            model_dir=tempfile.mkdtemp(),
            feature_columns=feature_columns
        )

    else:
        raise Exception('evaluate_model: no model_type set')

    start = timer()
    if split_type == cf.FLAGS['SEVENTY_THIRTY']:
        model.train(input_fn=training_input_fn)
        y_pred_dict = model.predict(input_fn=testing_input_fn)
        y_pred_list = []
        for pred in y_pred_dict:
            y_pred_list.append(pred.get('predictions')[0])

        return y_pred_list

    elif split_type == cf.FLAGS['TEN_FOLD_CROSS']:
        for index, value in enumerate(training_input_fn):
            model.train(input_fn=value)
            results = model.evaluate(input_fn=testing_input_fn[index])

            print('10-Fold-Cross Iteration {}'.format(index + 1))
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))

    else:
        raise Exception('evaluate_model: no split_type set')

    end = timer()
    print('Time taken: {}'.format(end - start))


def mean_square_error(y_pred, y_true):
    mse = 0
    n_instances = len(y_pred)
    for pred, truth in zip(y_pred, y_true):
        mse += (pred - truth)**2

    print(mse / n_instances)


main()
