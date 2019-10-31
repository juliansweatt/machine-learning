import numpy
import math
import matplotlib.pyplot as plt
import cvxpy as cp
import randomSet as util

def plotPoints(set1, set2, w, b):
    x1, y1 = set1.T
    x2, y2 = set2.T
    plt.plot(x1, y1, '.', color='b')
    plt.plot(x2, y2, '.', color='r')
    plt.axis('equal')
    print(w," |||| ", b)

def plotHyperplane(w,b,minX,maxX):
    # Hyperplane 
    x_points = numpy.array(range(minX, maxX))
    y_points =[]
    for x in x_points:
        y_points.append((-1 * b - w[0]*x)/w[1])

    # Soft Margin
    x_soft_upper = []
    y_soft_upper = []
    x_soft_lower = []
    y_soft_lower = []
    sm = 1/w
    for i,x in enumerate(x_points):
        x_soft_upper.append(x + sm[0])
        y_soft_upper.append(y_points[i] + sm[0])
        x_soft_lower.append(x - sm[0])
        y_soft_lower.append(y_points[i] - sm[0])

    plt.plot(x_soft_upper,y_soft_upper, color='grey')
    plt.plot(x_soft_lower,y_soft_lower, color='grey')
    plt.plot(x_points,y_points, color='purple')

def optimization(X, Y, wDim):
    # Setup SVM Optimization Problem
    w = cp.Variable((wDim,1))
    b = cp.Variable()
    epsi = cp.Variable((len(X),1))
    half = cp.Constant(1/2)
    reg = cp.square(cp.norm(w,2)) # Should square
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
    return {'w': w.value, 'b': b.value}

if __name__ == '__main__':
    randomSet1, randomSet2, X, Y = util.myPoints()
    optimals = optimization(X,Y, 2)
    plotPoints(randomSet1, randomSet2, optimals['w'], optimals['b'])
    plotHyperplane(optimals['w'], optimals['b'], int(min(X)), int(max(X)))
    plt.show()