# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:17:28 2016

@author: zz
"""

import pickle
import numpy as np
import random

def LoadPickle(filename):
    try:
        with open(filename, 'rb') as fTample:
            tamples = pickle.load(fTample)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
    return tamples


class BPNN:
    def __init__(self, x_num, delta3_num, delta2_num, y_num):
        self.delta3 = np.asmatrix(np.zeros(delta3_num))
        self.net3 = np.asmatrix(np.zeros(delta3_num))
        self.b3 = np.asmatrix(np.zeros(delta3_num))
        self.delta2 = np.asmatrix(np.zeros(delta2_num))
        self.net2 = np.asmatrix(np.zeros(delta2_num))
        self.b2 = np.asmatrix(np.zeros(delta2_num))
        self.y = np.asmatrix(np.zeros(y_num))
        self.net1 = np.asmatrix(np.zeros(y_num))
        self.b1 = np.asmatrix(np.zeros(y_num))

        self.W3 = np.random.uniform(-0.1, 0.1, size=delta3_num*x_num)
        self.W3 = np.asmatrix(self.W3.reshape((delta3_num, x_num)))

        self.W2 = np.random.uniform(-0.1, 0.1, size=delta2_num*delta3_num)
        self.W2 = np.asmatrix(self.W2.reshape((delta2_num, delta3_num)))

        self.W1 = np.random.uniform(-0.1, 0.1, size=y_num*delta2_num)
        self.W1 = np.asmatrix(self.W1.reshape((y_num, delta2_num)))

        self.deltaW3 = np.asmatrix(np.zeros(delta3_num*x_num).reshape((delta3_num, x_num)))
        self.deltaW2 = np.asmatrix(np.zeros(delta2_num*delta3_num).reshape((delta2_num, delta3_num)))
        self.deltaW1 = np.asmatrix(np.zeros(y_num*delta2_num).reshape((y_num, delta2_num)))

        self.deltab3 = np.asmatrix(np.zeros(delta3_num))
        self.deltab2 = np.asmatrix(np.zeros(delta2_num))
        self.deltab1 = np.asmatrix(np.zeros(y_num))

        self.partia_E_net3 = np.asmatrix(np.zeros(delta3_num))
        self.partia_E_net2 = np.asmatrix(np.zeros(delta2_num))
        self.partia_E_net1 = np.asmatrix(np.zeros(y_num))
        
    def GetSigmod(self, z):
        return 1/(1+np.exp(-z))

    def GetNet(self, W, b, x):
        y = np.dot(np.asmatrix(W), np.asmatrix(x).T)
        y = y + np.asmatrix(b).T
        return y.T

    def PredictSingleSample(self, x):
        self.net3 = self.GetNet(self.W3, self.b3, x)
        self.delta3 = self.GetSigmod(self.net3)
#        print self.delta3

        self.net2 = self.GetNet(self.W2, self.b2, self.delta3)
        self.delta2 = self.GetSigmod(self.net2)
#        print self.delta2

        self.net1 = self.GetNet(self.W1, self.b1, self.delta2)
        self.y = self.GetSigmod(self.net1)

        return self.y

    def GetSingleLoss(self, y):
        return np.sum(np.power(self.y-y, 2))

    def Partial_E_y(self, y):
        return self.y - y

    def GetSigmodDerivative(self, delta):
        return np.asarray(delta)* np.asarray(1-delta)

    def GetPartial_E_net1(self, y):
        partial_E_y = self.Partial_E_y(y)
        sigmodDerivative = self.GetSigmodDerivative(self.y)
        partial_E_net1 = np.asarray(partial_E_y)*np.asarray(sigmodDerivative)
        return partial_E_net1

    def GetNablaW1_b1(self, y):
        partial_E_net1 = self.GetPartial_E_net1(y)
#        print partial_E_net1,'aaaa\n'
#        print self.delta2
        self.deltab1 = np.dot(partial_E_net1, 1)
        self.deltaW1 = np.dot(np.asmatrix(partial_E_net1).T, np.asmatrix(self.delta2))
        self.partial_E_net1 = partial_E_net1
        
#        print self.deltab1,self.deltaW1,'bbbb\n'
        return self.deltaW1, self.deltab1

    def GetPartial_E_delta2(self):
        partia_E_net1 = self.partial_E_net1
        return np.dot(partia_E_net1, np.asmatrix(self.W1))

    def GetPartial_E_net2(self):
        partial_E_delta2 = self.GetPartial_E_delta2()
        sigmodDerivative = self.GetSigmodDerivative(self.delta2)

        partial_E_net2 = np.asarray(partial_E_delta2)* np.asarray(sigmodDerivative)
        return partial_E_net2

    def GetNablaW2_b2(self):
        partial_E_net2 = self.GetPartial_E_net2()
        self.deltab2 = np.dot(partial_E_net2, 1)
        self.deltaW2 = np.dot(np.asmatrix(partial_E_net2).T, np.asmatrix(self.delta3))
        self.partial_E_net2 = partial_E_net2
        return self.deltaW2, self.deltab2

    def GetPartial_E_delta3(self):
        partia_E_net2 = self.partial_E_net2
        return np.dot(partia_E_net2, np.asmatrix(self.W2))

    def GetPartial_E_net3(self):
        partial_E_delta3 = self.GetPartial_E_delta3()
        sigmodDerivative = self.GetSigmodDerivative(self.delta3)

        partial_E_net3 = np.asarray(partial_E_delta3)*np.asarray(sigmodDerivative)
        return partial_E_net3

    def GetNablaW3_b3(self, x):
        partial_E_net3 = self.GetPartial_E_net3()
        self.deltab3 = np.dot(partial_E_net3, 1)
        self.deltaW3 = np.dot(np.asmatrix(partial_E_net3).T, np.asmatrix(x))
        self.partial_E_net3 = partial_E_net3
        return self.deltaW3, self.deltab3

    def TrainSingleSample(self, x, y, alpha):
        self.PredictSingleSample(x)

        self.GetNablaW1_b1(y)
        self.GetNablaW2_b2()
        self.GetNablaW3_b3(x)

#        print self.deltaW1[0,0]
        self.b1 -= alpha*self.deltab1
        self.b2 -= alpha*self.deltab2
        self.b3 -= alpha*self.deltab3
        self.W1 -= alpha*self.deltaW1
        self.W2 -= alpha*self.deltaW2
        self.W3 -= alpha*self.deltaW3

        return self.GetSingleLoss(y)

    def Train(self, X, Y, alpha, e, times):
        for time in range(times):
            loss = 0
            for x, y in zip(X, Y):
                a = random.randint(1, 4)
#                if a != 2:
#                    continue
                loss += self.TrainSingleSample(x, y, alpha)
            loss /= len(X)
            print time,loss
            if loss < e:
                break

#a = BPNN(10, 5, 3, 2)
#X = np.array([[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1]])
#Y = np.array([[1,0],[0,1]])
#a.Train(X,Y,1,0.00003,100000)
#print a.PredictSingleSample(X[0])   

if __name__ == "__main__":
    try:
        with open("tample.txt", 'rb') as fTample :
            A = pickle.load(fTample)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
        
    random.shuffle(A)
    
    Y = A[:, :9]
    X = A[:, 9:]

    a = BPNN(96, 18, 32, 9)
    a.Train(X, Y, 0.3, 0.0003, 10000)

    with open('BP.pickle', 'rb') as BP:
        Bp = pickle.load(BP)
    print Bp.PredictSingleSample(A[0, 9:])
