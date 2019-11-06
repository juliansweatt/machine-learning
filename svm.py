import numpy
import math
import matplotlib.pyplot as plt
import cvxpy as cp
from common import Util, RandomData

class SVM():
    def __init__(self, classA=False, classB=False, *, C=1, plotNow=False, printReport=False, brute_loo=True, brute_mute=False):
        # Initialize Variables and Parameters
        self.optimized = False
        self.plotPrepared = False
        self.constantC = C
        self.supportVectors = []
        self.margin = 0
        self.printReport = printReport
        self.brute = brute_loo
        self.brute_mute = False

        # Generate Random Data or Accept Provided Sets
        if classA and classB:
            self.classA = classA
            self.classB = classB
            self.fullSet = Util.combineSets(self.classA, self.classB)
            self.fullSetX, self.fullSetY = self.fullSet.T
        else:
            self.classA, self.classB, self.fullSetX, self.fullSetY, self.fullSet = RandomData.random_data()
            self.labels = RandomData.linear_labels()

        # Dimension of Class Data
        self.dimensions = len(self.classA[len(self.classA)-1])

        # Optimize and Plot or Report if Needed
        if plotNow or printReport:
            self.optimize()
            if plotNow:
                print("Graphing SVM, Report Will Generate After Graph Is Closed If Requested.")
                self.plot()
            if printReport:
                self.preparePlot()
                print(self)

    def optimize(self):
        self.optimals = self.train(self.fullSet, self.labels)
        self.optimized = True

    def plot(self):
        if not self.plotPrepared:
            self.preparePlot()
        plt.show()
    
    def preparePlot(self):
        if self.optimized and not self.plotPrepared:
            # Plot Each Portion
            self.plotHyperplane()
            self.plotSupportVectors()
            self.plotPoints()

            # Setup Plot
            plt.title('Support Vector Machine')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(pad=2)
        else:
            if not self.optimized:
                print("SVM must be optimized to plot")
        self.plotPrepared = True

    def plotPoints(self):
        x1, y1 = self.classA.T
        x2, y2 = self.classB.T
        plt.plot(x1, y1, '.', color='b', label='Class A')
        plt.plot(x2, y2, '.', color='r', label='Class B')
        plt.axis('equal')

    def classify(self, x, w, b):
        y = self.predict(x,w,b)
        return Util.sign(y)

    def predict(self, x, w, b):
        wTop = w[:(len(w)-1)]
        wTop = wTop[0]
        wBottom = w[len(w)-1]
        return (-1 * b - sum(wTop*x))/wBottom

    def plotHyperplane(self):
        w = self.optimals['w']
        b = self.optimals['b']

        # Hyperplane 
        x_points = numpy.linspace(int(min(self.fullSetX)), int(max(self.fullSetX)), len(self.fullSet))
        y_points =[]
        for x in x_points:
            y_points.append(self.predict(x,w,b))
        plt.plot(x_points,y_points, color='purple', label='Classifier')

        # Soft Margin
        self.margin = (1/Util.norm(w))

        # Plot Margins (Offset Based)
        soft_upper = []
        soft_lower = []
        for wNum, wVal in enumerate(w):
            offset_distance = (1/wVal)/2
            upper_bounds = []
            lower_bounds = []
            for i,x in enumerate(x_points):
                if wNum == 0:
                    upper_bounds.append(x + offset_distance)
                    lower_bounds.append(x - offset_distance)
                elif wNum == 1:
                    upper_bounds.append(y_points[i] + offset_distance)
                    lower_bounds.append(y_points[i] - offset_distance)
            soft_upper.append(upper_bounds)
            soft_lower.append(lower_bounds)

        plt.plot(soft_upper[0],soft_upper[1], linestyle=':', color='grey', label='Margin')
        plt.plot(soft_lower[0],soft_lower[1], linestyle=':', color='grey')

    def calculateSupportVectors(self):
        self.supportVectors = []
        for i,constraint in enumerate(self.optimals['constraints'][:len(self.fullSet)]):
            for dual in constraint.dual_value:
                if not numpy.isclose(dual, 0) and dual >= 0:
                    self.supportVectors.append((self.fullSetX[i], self.fullSetY[i]))
        return self.supportVectors

    def plotSupportVectors(self):
        if len(self.supportVectors) < 1:
            self.calculateSupportVectors()

        if len(self.supportVectors):
            x_sv, y_sv = numpy.asarray(self.supportVectors).T
            plt.plot(x_sv, y_sv, 'o', color='orange', label='Support Vector')
        else:
            print("No Support Vectors Found.")

    def train(self, full_set, labels):
        # Setup SVM Optimization Problem
        w = cp.Variable((self.dimensions,1))
        b = cp.Variable()
        epsi = cp.Variable((len(full_set),1))
        half = cp.Constant(1/2)
        reg = cp.square(cp.pnorm(w,2))
        c = cp.Constant(self.constantC)
        loss = cp.norm(epsi,1)

        # Setup Optimization Constraints
        constraints = []
        numPoints = len(full_set)
        for i in range(numPoints):
            constraints += [labels[i] * (Util.dot(w, full_set[i]) + b) >= 1 - epsi[i]]
        for i in range(numPoints):
            constraints += [epsi[i]>=0]

        # Solve Problem 
        prob = cp.Problem(cp.Minimize(half * reg + c*loss), constraints)
        prob.solve()

        # Return Optimal w and b Values
        return {'w': w.value, 'b': b.value, 'constraints': constraints, 'epsi': epsi.value}

    def verify_leave_one_out_error(self,numSupportVectors, numVectors):
        return 100*numSupportVectors/numVectors 

    def leave_one_out_error(self):
        print("Please Wait, Calculating Leave One Out Error...")
        correct = 0
        for i, pair in enumerate(self.fullSet):
            if not self.brute_mute:
                print("LOO Progress:",i+1,"/",len(self.fullSet))
            x = pair[0]
            known_label = self.labels[i]
            optimal = None
            if i == 0:
                # Begining
                optimal = self.train(self.fullSet[1:], self.labels[1:])
            elif i == len(self.fullSetX)-1:
                # End
                optimal = self.train(self.fullSet[:len(self.fullSetX)-1], self.labels[:len(self.fullSetX)-1])
            else:
                # Middle
                optimal = self.train((list(self.fullSet[:i]) + list(self.fullSet[i+1:])),(list(self.labels[:i]) + list(self.labels[i+1:])))

            if self.classify(x,optimal.get('w'),optimal.get('b')) == known_label:
                correct += 1
        
        num_vectors = len(self.labels)
        mistakes = num_vectors - correct
        return 100*(mistakes/num_vectors)

    def __str__(self):
        output = str()
        if self.brute:
            output += 'Leave One Out Error: '
            output += str(self.leave_one_out_error()) + "%"
            output += '\n'
        output += 'Support Vector Ratio: '
        output += str(self.verify_leave_one_out_error(len(self.supportVectors), len(self.fullSet))) + "%"
        output += '\n'
        output += 'Constant C: '
        output += str(self.constantC)
        output += '\n'
        output += 'Margin: '
        output += str(self.margin)
        return output