import numpy.random
import cvxpy as cp
import math

class Util():
    @staticmethod
    def combineSets(a,b):
        return numpy.concatenate((a,b), axis=0)

    @staticmethod
    def dot(w, cartisian, dimensions=2):
        products = list()
        for i in range(dimensions):
            products.append(cp.multiply(w[i], cartisian[i]))
        return cp.sum(products)

    @staticmethod
    def norm(U):
        normalization = 0.0
        for u in U:
            normalization += u**2
        normalization = math.sqrt(normalization)
        return normalization

    @staticmethod
    def sign(number):
        if number < 0:
            return 1
        if number >= 0:
            return -1

class RandomData():
    @staticmethod
    def __random_gauss__(mean, covMatrix, size=200):
        numpy.random.seed(seed=4789)
        return numpy.random.multivariate_normal(mean, covMatrix, size)
    
    @staticmethod
    def random_data(size=200):
        mean1 = (-1, -1)
        mean2 = (1, 1)
        covariance = [
            [1,0],
            [0,1]
        ]
        randomSet1 = RandomData.__random_gauss__(mean1, covariance, size)
        randomSet2 = RandomData.__random_gauss__(mean2, covariance, size)
        fullSet = Util.combineSets(randomSet1, randomSet2)
        X, Y = fullSet.T
        return randomSet1, randomSet2, X, Y, fullSet

    @staticmethod
    def log_labels(size=400):
        labels = []
        mid_point = size / 2
        for i in range(size):
            lab = 0
            if i < mid_point:
                labels.append(0)
            else:
                labels.append(1)
        return labels

    @staticmethod
    def linear_labels(size=400):
        labels = []
        mid_point = size / 2
        for i in range(size):
            lab = 0
            if i < mid_point:
                labels.append(-1)
            else:
                labels.append(1)
        return labels