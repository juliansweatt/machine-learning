import randomSet as util
import matplotlib.pyplot as plt
import math
import numpy

def linear_regression(X, Y):
    x_mean = numpy.mean(X)
    y_mean = numpy.mean(Y)
    numerator = 0
    denominator = 0
    for i in range(len(X)):
        numerator += (X[i]-x_mean)*(Y[i]-y_mean)
        denominator += (X[i] - x_mean)**2
    
    # m = b1, b = b0
    b1 = numerator/denominator
    b0 = y_mean - b1 * x_mean

    # Return Linear Regression Coefficients
    return b1, b0

def plot_line(X,m,b):
    y_values = []
    for x in X:
        y_values.append(linear_prediction(x,m,b))
    plt.plot(X, y_values, color='green')

def linear_prediction(x,m,b):
    return m*x+b

def binary_classifier(x,m,b, bound=0):
    if linear_prediction(x,m,b) <= bound:
        return -1
    else:
        return 1

def leave_one_out_error(X,Y,labels):
    correct = 0
    for i, x in enumerate(X):
        known_label = labels[i]
        known_x = X[i]
        b = 0
        m = 0
        if i == 0:
            # Begining
            b,m = linear_regression(X[1:], Y[1:])
        elif i == len(X)-1:
            # End
            b,m = linear_regression(X[:len(X)-1], Y[:len(X)-1])
        else:
            # Middle
            b,m = linear_regression((list(X[:i]) + list(X[i+1:])),(list(Y[:i]) + list(Y[i+1:])))


        if binary_classifier(x,b,m) == labels[i]:
            correct += 1
    return 1-(correct/len(labels)) # TODO Check Accuracy

def plot_points(set, colorCode):
    x1, y1 = set.T
    plt.plot(x1, y1, '.', color=colorCode)
    plt.axis('equal')

if __name__ == '__main__':
    set_1, set_2, X, Y, full_set, labels = util.myPoints()
    # plot_points(full_set, 'b')
    plot_points(set_1, 'b')
    plot_points(set_2, 'r')
    m,b = linear_regression(X,Y)
    plot_line(X, m, b)
    print("LOO Error",str(leave_one_out_error(X,Y,labels)) + "%")
    plt.show()