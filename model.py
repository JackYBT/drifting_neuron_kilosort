from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.utils import class_weight
from sklearn.ensemble import IsolationForest

import numpy as np

from uni_multi_variate_helpers import extract_template, return_utilized_channels, convert_spike_time_to_time_bin_count
import time


def train_model(config):
    # Identifying the utilized_channels
    cur_template = extract_template(
        template_used_data_path=config["template_used_data_path"], templateId=config["NEURON_ID"])
    utilized_channels = np.array(return_utilized_channels(cur_template))

    X = []
    y = []
    print("config['selected_trials']", config['selected_trials'])
    for trial in config['selected_trials']:
        curTrial = config['selected_trials_spikes_fr_voltage'][trial]
        if len(X) == 0:
            X = curTrial['univariate_projection'][:, utilized_channels]
        else:
            X = np.concatenate(
                [X, curTrial['univariate_projection'][:, utilized_channels]])

        kilosort_spike_time = curTrial["kilosort_spike_time"]
        kilosort_spike_bin = convert_spike_time_to_time_bin_count(
            kilosort_spike_time, trial, config)
        if len(y) == 0:
            y = kilosort_spike_bin
        else:
            y = np.concatenate([y, kilosort_spike_bin])

        # assert len(X) == len(y)
    print("X", len(X), X)
    print("y", len(y), y)

    print("num of positive integers", np.count_nonzero(y == 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41)

    # multivariate_NN_model(config, X_train, X_test, y_train, y_test)
    isolation_forest_model(config, X_train, X_test, y_train, y_test)


def multivariate_NN_model(config, X_train, X_test, y_train, y_test):

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    # model.add(Dense(16, activation='relu'))  # New added layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(learning_rate=0.001),
                  'binary_crossentropy', metrics=['AUC'])

    start_time = time.time()

    # Training for one epoch
    model.fit(X_train, y_train, class_weight=class_weights,
              epochs=1, batch_size=32)

    end_time = time.time()

    training_time_for_one_epoch = end_time - start_time

    print("Training time for one epoch: {}".format(training_time_for_one_epoch))

    # Continue with the rest of your epochs
    model.fit(X_train, y_train, class_weight=class_weights,
              epochs=200, batch_size=32)

    y_pred = model.predict(X_test)
    print("original y_pred", y_pred)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print("y_pred", y_pred)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # False Positive Rate
    false_positive_rate = fp / (fp + tn)

    # False Negative Rate
    false_negative_rate = fn / (fn + tp)

    # True Positive Rate
    true_positive_rate = tp / (tp + fn)

    # True Negative Rate
    true_negative_rate = tn / (tn + fp)

    print("Accuracy: {}".format(accuracy))
    print("False Positive Rate: {}".format(false_positive_rate))
    print("False Negative Rate: {}".format(false_negative_rate))
    print("True Positive Rate: {}".format(true_positive_rate))
    print("True Negative Rate: {}".format(true_negative_rate))

    return model, rmse, y_pred, y_test


def multivariate_regression(config, X_train, X_test, y_train, y_test):

    # creating Linear Regression model
    model = LogisticRegression(class_weight='balanced', max_iter=5000)

    # training the model with training data
    model.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred = model.predict(X_test)

    seen = []
    for elem in y_pred:
        if elem not in seen:
            seen.append(elem)
    for elekm in y_test:
        if elekm not in seen:
            seen.append(elekm)
    print('seen', seen)
    # Calculating the root mean squared error
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    conf_mat = confusion_matrix(y_test, y_pred)
    if conf_mat.size > 4:  # This means it's a multiclass problem
        # True Positives for each class are the diagonal elements
        tp = np.diag(conf_mat)
        fp = np.sum(conf_mat, axis=0) - tp  # False Positives for each class
        fn = np.sum(conf_mat, axis=1) - tp  # False Negatives for each class
        tn = np.sum(conf_mat) - (fp + fn + tp)  # True Negatives for each class
    else:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy: ", accuracy)

    # False Positive Rate
    false_positive_rate = fp / (fp + tn)
    print("False Positive Rate: ", false_positive_rate)

    # False Negative Rate
    false_negative_rate = fn / (fn + tp)
    print("False Negative Rate: ", false_negative_rate)

    # True Positive Rate
    true_positive_rate = tp / (tp + fn)
    print("True Positive Rate: ", true_positive_rate)

    # True Negative Rate
    true_negative_rate = tn / (tn + fp)
    print("True Negative Rate: ", true_negative_rate)

    return model, rmse, y_pred, y_test


def isolation_forest_model(config, X_train, X_test, y_train, y_test):

    # Define the model
    clf = IsolationForest(contamination=0.01, random_state=0)

    # Fit the model
    start_time = time.time()
    clf.fit(X_train)
    end_time = time.time()

    training_time_for_one_epoch = end_time - start_time
    print("Training time for one epoch: {}".format(training_time_for_one_epoch))

    # Predict the anomalies in the data
    # The method returns 1 for inliers and -1 for outliers.
    y_pred = clf.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # False Positive Rate
    false_positive_rate = fp / (fp + tn)

    # False Negative Rate
    false_negative_rate = fn / (fn + tp)

    # True Positive Rate
    true_positive_rate = tp / (tp + fn)

    # True Negative Rate
    true_negative_rate = tn / (tn + fp)

    print("Accuracy: {}".format(accuracy))
    print("False Positive Rate: {}".format(false_positive_rate))
    print("False Negative Rate: {}".format(false_negative_rate))
    print("True Positive Rate: {}".format(true_positive_rate))
    print("True Negative Rate: {}".format(true_negative_rate))

    return clf, rmse, y_pred, y_test
