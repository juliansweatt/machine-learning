import numpy.random

def combineSets(a,b):
    return numpy.concatenate((a,b), axis=0)

def random2DGauss(mean, covMatrix):
    numpy.random.seed(seed=4789)
    return numpy.random.multivariate_normal(mean, covMatrix, 200)

def myPoints():
    mean1 = (-1, -1)
    mean2 = (1, 1)
    covarianceMatrix = [
        [1,0],
        [0,1]
    ]
    randomSet1 = random2DGauss(mean1, covarianceMatrix)
    randomSet2 = random2DGauss(mean2, covarianceMatrix)
    fullSet = combineSets(randomSet1, randomSet2)
    X, Y = fullSet.T

    labels = []
    for i in range(len(fullSet)):
        lab = 0
        if i < 200:
            labels.append(-1)
        else:
            labels.append(1)
    return randomSet1, randomSet2, X, Y, fullSet, labels