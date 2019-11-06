from common import Util, RandomData
import matplotlib.pyplot as plt
import cvxpy as cp
import math
import numpy

class Regression():
    def __init__(self, X=None, Y=None, class_labels=None):
        if X and Y and class_labels:
            # Existing Points
            self.X = X
            self.Y = Y
            self.labels = class_labels
            self.class_a = None
            self.class_b = None
        else:
            # Generate Points
            set_1, set_2, X_values, Y_values, full_set = RandomData.random_data()
            labels = RandomData.linear_labels()
            self.X = X_values
            self.Y = Y_values
            self.class_a = set_1
            self.class_b = set_2
            self.labels = labels
            self.full_set = full_set

    def run(self, calculate_error=True, plot_now=True):
        if calculate_error:
            error = self.leave_one_out_error()
            print("LOO Error",str(error) + "%")
        if plot_now:
            self.plot()

    def plot_points(self,set, colorCode, label):
        x1, y1 = set.T
        plt.plot(x1, y1, '.', color=colorCode, label=label)
        plt.axis('equal')

    def plot(self):
        # Regression-Specific
        raise NotImplementedError

    def classifier(self,X=None, Y=None):
        # Regression-Specific
        raise NotImplementedError 

    def plot_line(self,X,m,b, color='green', label=None):
        # Regression-Specifc
        raise NotImplementedError

    def prediction(self,x,m,b):
        # Regression-Specific
        raise NotImplementedError 

    def classify(self,x,m,b, bound=0):
        # Regression-Specific
        raise NotImplementedError

    def leave_one_out_error(self):
        # Regression-Specific
        raise NotImplementedError

class LinearRegression(Regression):
    def __init__(self,X=None, Y=None, labels=None, calculate_error=False, plot_now=False):
        Regression.__init__(self,X=X,Y=Y,class_labels=labels)
        self.run(plot_now=plot_now,calculate_error=calculate_error)

    def plot(self):
        # Plot Points
        if self.class_a.any() and self.class_b.any():
            self.plot_points(self.class_a, 'b', label='Class A')
            self.plot_points(self.class_b, 'r', label='ClassB')
        else:
            self.plot_points(zip(self.X, self.Y), 'b', label="Full Set")
        
        # Plot Regressions
        slope, b = self.classifier()
        self.plot_line(self.X, slope, b, color='grey', label="Traditional Regression")
        self.plot_line(self.X, -slope, b, label="Linear Classifier")

        # Setup Plot
        plt.title('Regression')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(pad=2)
        plt.show()

    def plot_line(self,X,m,b, color='green', label=None):
        y_values = []
        for x in X:
            y_values.append(self.prediction(x,m,b))
        plt.plot(X, y_values, color=color, label=label)

    def classify(self,x,m,b, bound=0):
        if self.prediction(x,m,b) <= bound:
            return -1
        else:
            return 1
    
    def classifier(self,X=None, Y=None):
        if type(X) != None and type(Y) != None:
            X = self.X
            Y = self.Y
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

    def prediction(self,x,m,b):
        return m*x+b

    def leave_one_out_error(self):
        correct = 0
        for i, x in enumerate(self.X):
            known_label = self.labels[i]
            b = 0
            m = 0
            if i == 0:
                # Begining
                b,m = self.classifier(self.X[1:], self.Y[1:])
            elif i == len(self.X)-1:
                # End
                b,m = self.classifier(self.X[:len(self.X)-1], self.Y[:len(self.X)-1])
            else:
                # Middle
                b,m = self.classifier((list(self.X[:i]) + list(self.X[i+1:])),(list(self.Y[:i]) + list(self.Y[i+1:])))

            if self.classify(x,b,m) == known_label:
                correct += 1
        return 100*(1-(correct/len(self.labels)))

class LogisticRegression(Regression):
    def __init__(self, X=None, Y=None, labels=None, calculate_error=False, plot_now=False):
        Regression.__init__(self,X=X, Y=Y, class_labels=labels)
        set_1, set_2, X_values, Y_values, full_set = RandomData.random_data()
        labels = RandomData.log_labels()
        self.X = X_values
        self.Y = Y_values
        self.class_a = set_1
        self.class_b = set_2
        self.labels = labels
        self.full_set = full_set

        self.run(plot_now=plot_now, calculate_error=calculate_error)

    def plot(self):
        # Plot Points
        if self.class_a.any() and self.class_b.any():
            self.plot_points(self.class_a, 'b', label='Class A')
            self.plot_points(self.class_b, 'r', label='ClassB')
        else:
            self.plot_points(zip(self.X, self.Y), 'b', label="Full Set")
        
        # Plot Regressions
        beta = self.classifier()
        self.plot_line(beta,min(self.X), max(self.X), color='grey', label="Regression")

        # Setup Plot
        plt.title('Logistic Regression')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(pad=2)
        plt.show()

    def plot_line(self,beta, domain_min, domain_max, points=1000, color='green', label=None):
        y_values = []
        X = numpy.linspace(domain_min, domain_max, points)
        for x in X:
            y_values.append(self.prediction(x,beta))
        plt.plot(X, y_values, color=color, label=label)

    def classify(self,x,beta, bound=0):
        if self.prediction(x,beta) <= bound:
            return 1
        else:
            return 0

    def classifier(self,X=None, Y=None):
        if type(X) != None and type(Y) != None:
            X = self.X
            Y = self.Y
        beta = self.__optimize__(self.full_set, self.labels)
        return beta

    def prediction(self, x, beta):
        return ((-1 * beta[0] * x)/beta[1])

    def customMul(self, labelsArr, pointsArr):
        prod = 0
        for i in range(len(pointsArr)):
            for a in pointsArr[i]:
                prod+= a*labelsArr[i]
        return prod

    def __optimize__(self, full_set, labels):
        # Citation: https://www.cvxpy.org/examples/machine_learning/logistic_regression.html
        n = 2
        beta = cp.Variable(n)
        log_likelihood = cp.sum(cp.multiply(labels, full_set @ beta) - cp.logistic(full_set @ beta))
        problem = cp.Problem(cp.Maximize(log_likelihood))
        problem.solve()

        return beta.value

    def leave_one_out_error(self):
        correct = 0
        numPoints = len(self.full_set)
        for i, x in enumerate(self.X):
            print("LOO Progress:", i+1, "/", numPoints)
            known_label = self.labels[i]
            beta = 0
            if i == 0:
                # Begining
                beta = self.__optimize__(self.full_set[1:], self.labels[1:])
            elif i == numPoints-1:
                # End
                beta = self.__optimize__(self.full_set[:numPoints-1], self.labels[:numPoints-1])
            else:
                # Middle
                beta = self.__optimize__((Util.combineSets(self.full_set[:i],self.full_set[i+1:])), Util.combineSets(self.labels[:i], self.labels[i+1:]))

            if self.classify(x,beta) == known_label:
                correct += 1
        
        incorrect = numPoints-correct
        return 100*(incorrect/numPoints)