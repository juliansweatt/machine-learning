import numpy
import math
import matplotlib.pyplot as plt
import cvxpy as cp
import randomSet as util

class SVM():
    def __init__(self, classA=False, classB=False, *, C=1, plotNow=False, printReport=False):
        self.optimized = False
        self.plotPrepared = False
        self.constantC = C
        self.supportVectors = []
        self.margins = []
        self.printReport = printReport
        if classA and classB:
            self.classA = classA
            self.classB = classB
            fullSet = util.combineSets(self.classA, self.classB)
            self.fullSetX, self.fullSetY = fullSet.T
        else:
            self.classA, self.classB, self.fullSetX, self.fullSetY = util.myPoints()

        self.dimensions = len(self.classA[len(self.classA)-1])

        if plotNow or printReport:
            self.optimize()
            if printReport:
                self.preparePlot()
                print(self)
            if plotNow:
                self.plot()

    def optimize(self):
        self.optimals = self.optimization()
        self.calculateSupportVectors()
        self.optimized = True

    def plot(self):
        if not self.plotPrepared:
            self.preparePlot()
        plt.show()
    
    def preparePlot(self):
        if self.optimized and not self.plotPrepared:
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
        x_points = numpy.linspace(int(min(self.fullSetX)), int(max(self.fullSetX)), len(self.fullSetX))
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
            self.margins.append(softMargin)
            for i,x in enumerate(x_points):
                x_soft_upper.append(x + softMargin)
                y_soft_upper.append(y_points[i] + softMargin)
                x_soft_lower.append(x - softMargin)
                y_soft_lower.append(y_points[i] - softMargin)
        plt.plot(x_soft_upper,y_soft_upper, linestyle=':', color='grey', label='Margin')
        plt.plot(x_soft_lower,y_soft_lower, linestyle=':', color='grey')

    def calculateSupportVectors(self):
        self.supportVectors = []
        for i,constraint in enumerate(self.optimals['constraints']):
            for dual in constraint.dual_value:
                if math.isclose(dual,0, abs_tol=0.1): # TODO Fix support vector tolerance
                    self.supportVectors.append((self.fullSetX[i],self.fullSetY[i]))
        return self.supportVectors

    def plotSupportVectors(self):
        if len(self.supportVectors):
            self.supportVectors = self.supportVectors
        else:
            self.calculateSupportVectors()
        x_sv, y_sv = numpy.asarray(self.supportVectors).T
        plt.plot(x_sv, y_sv, '+', color='cyan', label='Support Vector')

    def optimization(self):
        # Setup SVM Optimization Problem
        w = cp.Variable((self.dimensions,1))
        b = cp.Variable()
        epsi = cp.Variable((len(self.fullSetX),1))
        half = cp.Constant(1/2)
        reg = cp.square(cp.norm(w,2))
        c = cp.Constant(self.constantC)
        loss = cp.sum(cp.pos(epsi))

        # Setup Optimization Constraints
        constraints = []
        for i in range(len(self.fullSetX)):
            constraints += [self.fullSetY[i] * (w * self.fullSetX[i] + b) >= 1 - epsi[i]]

        # Solve Problem 
        prob = cp.Problem(cp.Minimize(half*reg + c * loss), constraints)
        prob.solve()

        # Return Optimal w and b Values
        return {'w': w.value, 'b': b.value, 'constraints': constraints}

    def leaveOneOutError(self,numSupportVectors, numVectors):
        return numSupportVectors/numVectors

    def __str__(self):
        output = str()
        if len(self.supportVectors)>0:
            output += 'Support Vectors: '
            output += str(self.supportVectors)
            output += '\n'
            output += 'Leave One Out Error: '
            output += str(self.leaveOneOutError(len(self.supportVectors), len(self.fullSetX)))
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