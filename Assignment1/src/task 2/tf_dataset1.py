#! usr/bin/python3
'''
Tensorflow dataset 1
'''

import tempfile
import config as cf

import pandas as pd
import tensorflow as tf


def main():
    CSV_COLUMNS = ['Instances', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4',
                   'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9',
                   'Feature_10', 'Target', 'Target_Class']
    # Read in full dataset
    df = pd.read_csv(
        cf.PATHS['SUMDATA_NOISELESS'],
        header=0,
        names=CSV_COLUMNS,
        sep=';'
    )

    # Split the dataset into training and testing
    df_training, df_testing = split_data_frame(df, cf.FLAGS['SEVENTY_THIRTY'])

    # Get the label columns, the output vectors in Tensor Flow terminology
    train_label_regression, test_label_regression = build_regression_labels(
        df_training,
        df_testing
    )

    # Build TF feature columns
    base_feature_columns = build_feature_columns()

    # TODO: implement 10 fold model training and evaluation

    # Build training and testing input function
    training_input_fn, testing_input_fn = build_regression_input_functions(
        df_training,
        df_testing,
        train_label_regression,
        test_label_regression
    )

    # Do that ML
    evaluate_model(
        base_feature_columns,
        cf.FLAGS['LINEAR_REGRESSION'],
        training_input_fn,
        testing_input_fn
    )


def split_data_frame(df, split_type):
    df_depth = int(df.shape[0])
    if split_type is cf.FLAGS['SEVENTY_THIRTY']:
        split = int(df_depth * 0.7)
        df_training = df.iloc[:split, 1:]
        df_testing = df.iloc[split:, 1:]
        return df_training, df_testing

    elif split_type is cf.FLAGS['TEN_FOLD_CROSS']:
        df_testing = []
        df_training = []
        split = int(df_depth / 10)
        for i in range(10):
            # Note there is a loss of accuracy here converting float to integer
            # The final cross validation set will not reach the final row but each set size is consistent
            test_range = list(range(i * split, (i + 1) * split))
            full_range = list(range(0, df_depth))
            training_range = list(set(full_range) - set(test_range))
            df_testing.append(df.iloc[test_range, 1:])
            df_training.append(df.iloc[training_range, 1:])

        return df_training, df_testing


def build_regression_labels(df_training, df_testing):
    return df_training['Target'], df_testing['Target']


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


def build_regression_input_functions(df_training, df_testing, training_label, testing_label):
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

    return training_input_fn, testing_input_fn


def evaluate_model(feature_columns, model_type, training_input_fn, testing_input_fn):
    if model_type is cf.FLAGS['LINEAR_REGRESSION']:
        model = tf.estimator.LinearRegressor(
            model_dir=tempfile.mkdtemp(),
            feature_columns=feature_columns
        )

    # TODO: implement other algorithms and add to config file
    model.train(input_fn=training_input_fn)
    results = model.evaluate(input_fn=testing_input_fn)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


main()
