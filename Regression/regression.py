import randomSet as util
import matplotlib.pyplot as plt
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
        error = self.leave_one_out_error()
        print("LOO Error",str(error) + "%")
        self.plot()

    def classifier(self,X=None, Y=None):
        # Regression-Specific
        raise NotImplementedError 

    def plot_line(self,X,m,b, color='green'):
        y_values = []
        for x in X:
            y_values.append(self.linear_prediction(x,m,b))
        plt.plot(X, y_values, color=color)

    def linear_prediction(self,x,m,b):
        return m*x+b

    def classify(self,x,m,b, bound=0):
        if self.linear_prediction(x,m,b) <= bound:
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

if __name__ == '__main__':
    # plot_points(full_set, 'b')
    L = LinearRegression()
    L.run()