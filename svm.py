import numpy as np
import numpy 
import cvxopt
import cvxopt.solvers
import time, threading
import matplotlib 
import weakref
import random


limit = 0

def kernel(x1, x2):
    linear = np.dot(x1, x2)
    return linear

class SupportVectorMachine(object):
    def __init__(self, kernel=kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: 
            self.C = float(self.C)

    def compute(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        if self.kernel == kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def supportVector(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.supportVector(X))


if __name__ == "__main__":
    import pylab 
   

    # Produce random data-set
    def randomizeData():
        rand1 = random.randrange(0, 5)      # essentially data grid size
        rand2 = random.randrange(0, 5)
        rand3 = random.randrange(0, 5)
        rand4 = random.randrange(0, 5)
        firstMean = np.array([rand1, rand2])
        secondMean = np.array([rand3, rand4])

        covariance = np.array([[1.5, 1.0], [1.0, 1.5]])

        # Gaussian distribution 
        X1 = np.random.multivariate_normal(firstMean, covariance, 100)
        X1 = np.random.multivariate_normal(firstMean, covariance, 100)

        Y1 = np.ones(len(X1)) # grid of ones

        X2 = np.random.multivariate_normal(secondMean, covariance, 100)
        
        Y2 = np.ones(len(X2)) * -1 # grid of -1's
        return X1, Y1, X2, Y2

    def train(X1, Y1, X2, Y2):
        learnX1 = X1[:90]
        learnY1 = Y1[:90]
        learnX2 = X2[:90]
        learnY2 = Y2[:90]
        learnX = np.vstack((learnX1, learnX2)) #make stack horizontal
        learnY = np.hstack((learnY1, learnY2)) #make stack horizontal
        return learnX, learnY

    def data(X1, Y1, X2, Y2):
        X1_test = X1[90:]
        y1_test = Y1[90:]
        X2_test = X2[90:]
        y2_test = Y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(learnX1, learnX2, clf):
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        pylab.plot(learnX1[:,0], learnX1[:,1], "ro")
        pylab.plot(learnX2[:,0], learnX2[:,1], "bo")
        pylab.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        a0 = -2; a1 = f(a0, clf.w, clf.b)
        b0 = 2; b1 = f(b0, clf.w, clf.b)
        pylab.plot([a0,b0], [a1,b1], "k")

        a0 = -2; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 2; b1 = f(b0, clf.w, clf.b, 1)
        pylab.plot([a0,b0], [a1,b1], "k--")

        a0 = -2; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 2; b1 = f(b0, clf.w, clf.b, -1)
        pylab.plot([a0,b0], [a1,b1], "k--")

        pylab.axis("tight")
        pylab.show()

    def createPlot(learnX1, learnX2, clf, percent):
        pylab.clf() 
        pylab.clf()
        pylab.cla()
        pylab.plot(learnX1[:,0], learnX1[:,1], "ro")
        pylab.plot(learnX2[:,0], learnX2[:,1], "bo")
        pylab.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="y")

        X1, X2 = np.meshgrid(np.linspace(-10,10,10), np.linspace(-10,10,10))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.supportVector(X).reshape(X1.shape)
        pylab.contour(X1, X2, Z, [0.0], colors='black', linewidths=1.5, origin='lower')
        pylab.contour(X1, X2, Z + 1, [0.0], colors='blue', linewidths=2, origin='lower')
        pylab.contour(X1, X2, Z - 1, [0.0], colors='red', linewidths=2, origin='lower')
        sign = "%"
        pylab.title("Percentage of correct predictions %d %%" % (percent))
        pylab.axis("tight")

        pylab.show()
    

    
    def createData():
        X1, Y1, X2, Y2 = randomizeData()
        learnX, learnY = train(X1, Y1, X2, Y2)
        X_test, y_test = data(X1, Y1, X2, Y2)
        clf = SupportVectorMachine(C=0.1)
        clf.compute(learnX, learnY)
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        correctNum = float(correct) 
        predict =  float(len(y_predict)) 
        percent = (correctNum/predict)*100
        createPlot(learnX[learnY==1], learnX[learnY==-1], clf , percent)

    def generator():
        try:
            global limit 
            limit = limit - 1
            now = time.time()
            threading.Timer(3, generator).start()
            later = now + 10
            #if (limit > 0):
            createData()        #indent and remove comments for limit  c
                #os._exit()
        except RuntimeError:
            pass    
    #limit = input("Please enter maximum number of graphs to generate: ")
    #limit = int(limit)+1
    generator()

