import numpy
import math
import matplotlib.pyplot as plt
import cvxpy as cp
import randomSet as util

def plotPoints(classA, classB, w, b):
    x1, y1 = classA.T
    x2, y2 = classB.T
    plt.plot(x1, y1, '.', color='b', label='Class A')
    plt.plot(x2, y2, '.', color='r', label='Class B')
    plt.axis('equal')
    print(w," |||| ", b)

def plotHyperplane(w,b,X,Y):
    # Hyperplane 
    x_points = numpy.linspace(int(min(X)), int(max(X)), len(X))
    y_points =[]
    for x in x_points:
        wTop = w[:(len(w)-1)]
        wTop = wTop[0]
        wBottom = w[len(w)-1]
        y_points.append((-1 * b - sum(wTop*x))/wBottom)
    plt.plot(x_points,y_points, color='purple', label='Classifier')

    # Soft Margin
    x_soft_upper = []
    y_soft_upper = []
    x_soft_lower = []
    y_soft_lower = []
    for wVal in w:
        softMargin = (1/wVal)
        for i,x in enumerate(x_points):
            x_soft_upper.append(x + softMargin)
            y_soft_upper.append(y_points[i] + softMargin)
            x_soft_lower.append(x - softMargin)
            y_soft_lower.append(y_points[i] - softMargin)
    plt.plot(x_soft_upper,y_soft_upper, linestyle=':', color='grey', label='Margin')
    plt.plot(x_soft_lower,y_soft_lower, linestyle=':', color='grey')

def plotSupportVectors(X, Y, constraints):
    supportVectors = []
    for i,constraint in enumerate(constraints):
        for dual in constraint.dual_value:
            if math.isclose(dual,0, abs_tol=0.1): # TODO Fix support vector tolerance
                supportVectors.append((X[i],Y[i]))
    x_sv, y_sv = numpy.asarray(supportVectors).T
    plt.plot(x_sv, y_sv, '+', color='cyan', label='Support Vector')
    return len(supportVectors)

def optimization(X, Y, wDim):
    # Setup SVM Optimization Problem
    w = cp.Variable((wDim,1))
    b = cp.Variable()
    epsi = cp.Variable((len(X),1))
    half = cp.Constant(1/2)
    reg = cp.square(cp.norm(w,2))
    c = cp.Constant(1)
    loss = cp.sum(cp.pos(epsi))

    # Setup Optimization Constraints
    constraints = []
    for i in range(len(X)):
        constraints += [Y[i] * (w * X[i] + b) >= 1 - epsi[i]]

    # Solve Problem 
    prob = cp.Problem(cp.Minimize(half*reg + c * loss), constraints)
    prob.solve()

    # Return Optimal w and b Values
    return {'w': w.value, 'b': b.value, 'constraints': constraints}

if __name__ == '__main__':
    randomSet1, randomSet2, X, Y = util.myPoints()
    optimals = optimization(X,Y, 2)
    plotPoints(randomSet1, randomSet2, optimals['w'], optimals['b'])
    plotHyperplane(optimals['w'], optimals['b'], X, Y)
    plotSupportVectors(X,Y,optimals['constraints'])
    plt.title('Support Vector Machine')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(pad=1)
    plt.show()