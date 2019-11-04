import numpy
import math
import matplotlib.pyplot as plt
import cvxpy as cp
import randomSet as util

class SVM():
    def __init__(self, classA=False, classB=False, *, C=1, plotNow=False, printReport=False):
        # Initialize Variables and Parameters
        self.optimized = False
        self.plotPrepared = False
        self.constantC = C
        self.supportVectors = []
        self.margins = []
        self.printReport = printReport

        # Generate Random Data or Accept Provided Sets
        if classA and classB:
            self.classA = classA
            self.classB = classB
            self.fullSet = util.combineSets(self.classA, self.classB)
            self.fullSetX, self.fullSetY = self.fullSet.T
        else:
            self.classA, self.classB, self.fullSetX, self.fullSetY, self.fullSet, self.labels = util.myPoints()

        # Dimension of Class Data
        self.dimensions = len(self.classA[len(self.classA)-1])

        # Optimize and Plot or Report if Needed
        if plotNow or printReport:
            self.optimize()
            if printReport:
                self.preparePlot()
                print(self)
            if plotNow:
                self.plot()

    def optimize(self):
        self.optimals = self.optimization()
        # self.calculateSupportVectors()
        self.optimized = True

    def plot(self):
        if not self.plotPrepared:
            self.preparePlot()
        plt.show()
    
    def preparePlot(self):
        if self.optimized and not self.plotPrepared:
            # Plot Each Portion
            self.plotPoints()
            self.plotHyperplane()
            self.plotSupportVectors()

            # Setup Plot
            plt.title('Support Vector Machine')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(pad=2)
        else:
            print("SVM must be optimized to plot")
        self.plotPrepared = True

    def plotPoints(self):
        x1, y1 = self.classA.T
        x2, y2 = self.classB.T
        plt.plot(x1, y1, '.', color='b', label='Class A')
        plt.plot(x2, y2, '.', color='r', label='Class B')
        plt.axis('equal')

    def plotHyperplane(self):
        w = self.optimals['w']
        b = self.optimals['b']

        # Hyperplane 
        x_points = numpy.linspace(int(min(self.fullSetX)), int(max(self.fullSetX)), len(self.fullSet))
        y_points =[]
        for x in x_points:
            wTop = w[:(len(w)-1)]
            wTop = wTop[0]
            wBottom = w[len(w)-1]
            y_points.append((-1 * b - sum(wTop*x))/wBottom)
        plt.plot(x_points,y_points, color='purple', label='Classifier')

        # Soft Margins
        soft_upper = []
        soft_lower = []
        for wNum, wVal in enumerate(w):
            softMargin = (1/wVal)
            self.margins.append(softMargin)
            upper_bounds = []
            lower_bounds = []
            for i,x in enumerate(x_points):
                if wNum == 0:
                    upper_bounds.append(x + softMargin)
                    lower_bounds.append(x - softMargin)
                elif wNum == 1:
                    upper_bounds.append(y_points[i] + softMargin)
                    lower_bounds.append(y_points[i] - softMargin)
            soft_upper.append(upper_bounds)
            soft_lower.append(lower_bounds)

        plt.plot(soft_upper[0],soft_upper[1], linestyle=':', color='grey', label='Margin')
        plt.plot(soft_lower[0],soft_lower[1], linestyle=':', color='grey')

    def calculateSupportVectors(self):
        self.supportVectors = []
        for i,constraint in enumerate(self.optimals['constraints'][:len(self.fullSet)]):
            for dual in constraint.dual_value:
                print("Dual",[i], "-", dual, dual+self.optimals['constraints'][i+400].dual_value)
                if dual >= 0 and dual <= 1: # TODO Fix support vector tolerance
                    self.supportVectors.append((self.fullSetX[i], self.fullSetY[i]))
        return self.supportVectors

    def plotSupportVectors(self):
        if len(self.supportVectors):
            self.supportVectors = self.supportVectors
        else:
            self.calculateSupportVectors()
        x_sv, y_sv = numpy.asarray(self.supportVectors).T
        plt.plot(x_sv, y_sv, '+', color='cyan', label='Support Vector')

    def dot(self, w, cartisian):
        products = list()
        for i in range(self.dimensions):
            products.append(cp.multiply(w[i], cartisian[i]))
        return cp.sum(products)

    def optimization(self):
        # Setup SVM Optimization Problem
        w = cp.Variable((self.dimensions,1))
        b = cp.Variable()
        epsi = cp.Variable((len(self.fullSet),1))
        half = cp.Constant(1/2)
        reg = cp.square(cp.pnorm(w,2))
        c = cp.Constant(self.constantC)
        loss = cp.norm(epsi,1)

        # Setup Optimization Constraints
        constraints = []
        numPoints = len(self.fullSet)
        for i in range(numPoints):
            constraints += [self.labels[i] * (self.dot(w, self.fullSet[i]) + b) >= 1 - epsi[i]] # TODO This should also be w DOT x
        for i in range(numPoints):
            constraints += [epsi[i]>=0]

        # Solve Problem 
        prob = cp.Problem(cp.Minimize(half * reg + c*loss), constraints)
        prob.solve()

        # Return Optimal w and b Values
        return {'w': w.value, 'b': b.value, 'constraints': constraints}

    def leaveOneOutError(self,numSupportVectors, numVectors):
        return numSupportVectors/numVectors # Actually recompute by leaving one out, retraining, and seeing if each point goes back correctly

    def __str__(self):
        output = str()
        if len(self.supportVectors)>0:
            output += 'Support Vectors: '
            output += str(self.supportVectors)
            output += '\n'
            output += 'Leave One Out Error: '
            output += str(self.leaveOneOutError(len(self.supportVectors), len(self.fullSet))) # Must actually calculate
            output += '\n'
        output += 'Contant C: '
        output += str(self.constantC)
        output += '\n'
        if len(self.margins)>0:
            output += 'Margins: '
            for margin in self.margins:
                output += str(margin[0])
                output += ' '
        return output