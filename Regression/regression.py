import randomSet as util
import matplotlib.pyplot as plt
import cvxpy as cvx
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
            set_1, set_2, X_values, Y_values, full_set, labels = util.myPoints()
            self.X = X_values
            self.Y = Y_values
            self.class_a = set_1
            self.class_b = set_2
            self.labels = labels
            ##############
            self.full_set = full_set
    
    def plot(self):
        # Plot Points
        if self.class_a.any() and self.class_b.any():
            self.plot_points(self.class_a, 'b')
            self.plot_points(self.class_b, 'r')
        else:
            self.plot_points(zip(self.X, self.Y), 'b')
        
        # Plot Regressions
        slope, b = self.classifier()
        self.plot_line(self.X, slope, b)
        plt.show()

    def run(self, calculate_error=True, plot_now=True):
        if calculate_error:
            error = self.leave_one_out_error()
            print("LOO Error",str(error) + "%")
        if plot_now:
            self.plot()

    def classifier(self,X=None, Y=None):
        # Regression-Specific
        raise NotImplementedError 

    def plot_line(self,X,m,b, color='green'):
        y_values = []
        for x in X:
            y_values.append(self.prediction(x,m,b))
        plt.plot(X, y_values, color=color)

    def prediction(self,x,m,b):
        # Regression-Specific
        raise NotImplementedError 

    def classify(self,x,m,b, bound=0):
        if self.prediction(x,m,b) <= bound:
            return -1
        else:
            return 1

    def leave_one_out_error(self):
        correct = 0
        for i, x in enumerate(self.X):
            known_label = self.labels[i]
            known_x = self.X[i]
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


            if self.classify(x,b,m) == self.labels[i]:
                correct += 1
        return 100*(1-(correct/len(self.labels))) # TODO Check Accuracy

    def plot_points(self,set, colorCode):
        x1, y1 = set.T
        plt.plot(x1, y1, '.', color=colorCode)
        plt.axis('equal')

class LinearRegression(Regression):
    def __init__(self,X=None, Y=None, labels=None):
        Regression.__init__(self,X=X,Y=Y,class_labels=labels)
    
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

class LogisticRegression(Regression):
    def __init__(self, X=None, Y=None, labels=None):
        Regression.__init__(self,X=X, Y=Y, class_labels=labels)

    def classifier(self,X=None, Y=None):
        if type(X) != None and type(Y) != None:
            X = self.X
            Y = self.Y
        w = self.__optimize__(X,Y, self.labels)
        # x_space = numpy.logspace(min(X), min(Y), len(X))
        # plt.plot(self.X, self.Y, '.', color='b')
        # plt.plot(x_space, w, color='purple')
        # plt.yscale('log')
        # plt.show()

    def prediction(self, w):
        print("predict")
        # (-w[0,0]*xp-w[2,0])/w[1,0]
        # return m*x+b

    def customMul(self, labelsArr, pointsArr):
        prod = 0
        for i in range(len(pointsArr)):
            for a in pointsArr[i]:
                prod+= a*labelsArr[i]
        return prod

    def __optimize__(self, X, Y, labels):
        # Citation: https://www.cvxpy.org/examples/machine_learning/logistic_regression.html
        n = len(X)
        # beta = cp.Variable(n)
        # lambd = cp.Parameter(nonneg=True)
        # log_likelihood = cp.sum(
        #     cp.multiply(self.Y, self.X @ beta) - cp.logistic(self.X @ beta)
        # )
        # problem = cp.Problem(cp.Maximize(log_likelihood/n - lambd * cp.norm(beta, 1)))
        # problem.solve()

        # Citation: http://niaohe.ise.illinois.edu/IE598_2016/logistic_regression_demo/
        w = cvx.Variable(n)
        lambd = cvx.Constant(1)
        loss = cvx.sum(cvx.logistic(-cvx.multiply(Y, X*w))) + lambd/2 * cvx.sum_squares(w)
        problem = cvx.Problem(cvx.Minimize(loss))
        problem.solve() 
        return w.value

        # http://i-systems.github.io/HSE545/iAI/ML/topics/05_Classification/11_Logistic_Regression.pdf
        # Xi = self.full_set
        # w = cvx.Variable([3,1])
        # # obj = cvx.Maximize(labels*Xi*w-cvx.sum(cvx.logistic(Xi*w)))
        # obj = cvx.Maximize(self.customMul(labels,Xi)*w-cvx.sum(cvx.logistic(Xi*w)))
        # prob = cvx.Problem(obj).solve()
        # print(w)
        # return w

if __name__ == '__main__':
    L = LinearRegression()
    L.run()

    _log = LogisticRegression()
    _log.classifier()
    # _log.run(calculate_error=False, plot_now=False)