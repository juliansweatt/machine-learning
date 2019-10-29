import numpy
import matplotlib.pyplot as plt
import cvxpy as cp

def combineSets(a,b):
    return numpy.concatenate((a,b), axis=0)

def arrayAtIndex(A, index):
    newA = []
    for a in A:
        newA.append(a[index])
    return newA

def random2DGauss(mean, covMatrix):
    numpy.random.seed(seed=4789)
    return numpy.random.multivariate_normal(mean, covMatrix, 200)

def plotPoints(set1, set2):
    x1, y1 = set1.T
    x2, y2 = set2.T
    plt.plot(x1, y1, '.', color='b')
    plt.plot(x2, y2, '.', color='r')
    plt.axis('equal')
    plt.show()

def optimization(X, Y):
    w = cp.Variable((2,1))
    b = cp.Variable()
    epsi = cp.Variable()
    half = cp.Constant(1/2)
    reg = cp.norm(w,2)
    c = cp.Constant(1)
    loss = cp.sum(epsi)
    prob = cp.Problem(cp.Minimize(half*reg + c * loss))
    return prob.solve()

if __name__ == '__main__':
    mean1 = (-1, -1)
    mean2 = (1, 1)
    covarianceMatrix = [
        [1,0],
        [0,1]
    ]
    randomSet1 = random2DGauss(mean1, covarianceMatrix)
    randomSet2 = random2DGauss(mean2, covarianceMatrix)
    fullSet = combineSets(randomSet1, randomSet2)
    X = arrayAtIndex(fullSet, 0)
    Y = arrayAtIndex(fullSet, 1)
    plotPoints(randomSet1, randomSet2)