import matplotlib.pyplot as plt

#Simple Linear Regression classifier and Plotting

def mean(values):
    return sum(values) / float(len(values))


def variance(values, mean):
    return sum([(x-mean)**2 for x in values])


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

def simple_linear_regression(train, test):
    y_predictions = list()
    x_predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        #predictions.append([row[0], yhat])
        y_predictions.append(yhat)
        x_predictions.append(row[0])
    return x_predictions, y_predictions


dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
testset = [[1], [2], [3], [4], [5]]
data = simple_linear_regression(dataset, testset)
plt.plot(data[0], data[1])
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
plt.show()


