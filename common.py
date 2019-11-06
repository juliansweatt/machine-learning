import numpy.random

class Util():
    @staticmethod
    def combineSets(a,b):
        return numpy.concatenate((a,b), axis=0)

class RandomData():
    @staticmethod
    def __random2DGauss__(mean, covMatrix, size=200):
        numpy.random.seed(seed=4789)
        return numpy.random.multivariate_normal(mean, covMatrix, size)

    @staticmethod
    def randomCartisian(size=200):
        mean1 = (-1, -1)
        mean2 = (1, 1)
        covarianceMatrix = [
            [1,0],
            [0,1]
        ]
        class_a = RandomData.__random2DGauss__(mean1, covarianceMatrix, size)
        class_b = RandomData.__random2DGauss__(mean2, covarianceMatrix, size)
        fullSet = Util.combineSets(class_a, class_b)
        X, Y = fullSet.T

        labels = []
        for i in range(len(fullSet)):
            lab = 0
            if i < 200:
                labels.append(-1)
            else:
                labels.append(1)
        return class_a, class_b, X, Y, fullSet, labels
    
    @staticmethod
    def randomLog(size=200):
        mean1 = (-1, -1)
        mean2 = (1, 1)
        covarianceMatrix = [
            [1,0],
            [0,1]
        ]
        randomSet1 = RandomData.__random2DGauss__(mean1, covarianceMatrix, size)
        randomSet2 = RandomData.__random2DGauss__(mean2, covarianceMatrix, size)
        fullSet = Util.combineSets(randomSet1, randomSet2)
        X, Y = fullSet.T

        labels = []
        for i in range(len(fullSet)):
            lab = 0
            if i < 200:
                labels.append(0)
            else:
                labels.append(1)
        return randomSet1, randomSet2, X, Y, fullSet, labels