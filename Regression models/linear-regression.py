import csv

from math import sqrt

def load_csv(filename):
    dataset = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        headers = reader.__next__()
        for row in reader:
            if len(row) != 2:
                continue
            for i in range(len(row)):
                row[i] = float(row[i].strip())
            dataset.append(row)
    return dataset

def rmse_metric(actual, predicted):
    error_sum = 0.0
    for i in range(len(actual)):
        difference = predicted[i] - actual[i]
        error_sum = error_sum + (difference ** 2)
    return sqrt(error_sum / len(actual))

def create_test_and_actual(test):
    actual = list()
    stripped = list()
    for row in test:
        c = list(row)
        actual.append(row[-1])
        c[-1] = None
        stripped.append(c)
    return stripped, actual

def evaluate(train, test, algorithm, *args):
    test_set, actual = create_test_and_actual(test)

    predicted = linear_regression(train, test_set, *args)
    rmse = rmse_metric(actual, predicted)
    
    print('actual:    ' + str(actual))
    print('predicted: ' + str(predicted))

    return rmse

def mean(values):
    return sum(values) / float(len(values))

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean = mean(x)
    y_mean = mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return b0, b1

def linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions

test = load_csv('../input/test.csv')
train = load_csv('../input/train.csv')

rmse = evaluate(train, test, linear_regression)
print('RMSE of linear regression:  ' + str(rmse))