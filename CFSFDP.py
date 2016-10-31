# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:50:20 2016

@author: zz
"""

import copy
import random as random
import pickle
import numpy as np
from matplotlib import pyplot as plt


def LoadPickle(filename):
    try:
        with open(filename, 'rb') as fTample:
            tamples = pickle.load(fTample)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
    return tamples


def GetPointDistance(x, y):
    return np.sqrt(np.sum(np.power(x-y, 2)))


def GetDistanceMatrix(tamples):
    tampleNum = tamples.shape[0]
    M = np.zeros((tampleNum, tampleNum))
    for i in range(tampleNum):
        for j in range(tampleNum):
            distance = GetPointDistance(tamples[i], tamples[j])
            print i, j
            M[i][j] = distance
            M[j][i] = distance
    return M


def GetRho(M, dmin):
    tampleNum = M.shape[0]
    rhos = np.zeros(tampleNum)
    for i in range(tampleNum):
        for j in range(tampleNum):
            if M[i][j] < dmin:
                rhos[i] += 1
    return rhos


def GetDelta(M, rhos):
    tampleNum = rhos.shape[0]
    deltas = np.zeros(tampleNum)
    minDistanceIndexs = np.zeros(tampleNum, dtype='int64')
    maxDistance = np.amax(M)
    for i in range(tampleNum):
        deltaTemp = float('inf')
        minDistanceIndexTemp = -1
        for j in range(tampleNum):
            if rhos[i] < rhos[j] or rhos[i] == rhos[j] and i < j:
                if(M[i][j] < deltaTemp):
                    deltaTemp = M[i][j]
                    minDistanceIndexTemp = j
        deltas[i] = min(deltaTemp, maxDistance)
        minDistanceIndexs[i] = minDistanceIndexTemp
    return deltas, minDistanceIndexs


def GetCenterPointSet(deltas, rhos):
    centerPoint = set()
    index = 0
    for delta, rho in zip(deltas, rhos):
        if rho > 8 and delta > 0.015:
            centerPoint.add(index)
        index += 1
    return centerPoint


def SetOtherPointLabel(minIndexs, tamples):
    unclassedPointNum = 1
    indexlist = None  # 中间变量保存
    while unclassedPointNum > 0:
        unclassedPointNum = 0
        for i, minIndex in enumerate(minIndexs):
            if i in centerPoint:
                minIndexs[i] = i
            elif minIndex in centerPoint:
                continue
            else:
                minIndexs[i] = minIndexs[minIndex]
                unclassedPointNum += 1

        if indexlist is None:
            indexlist = copy.copy(minIndexs)
        else:
            indexlist = np.vstack((indexlist, minIndexs))
    labelList = []
    for index in minIndexs:
        labelList.append(tamples[index][0])
    return indexlist, labelList


if __name__ == "__main__":
    tamples = LoadPickle("tample.txt")
#    M = GetDistanceMatrix(tamples[:, 1:])
#    with open("M.txt", 'wb') as Mtxt:
#        pickle.dump(M, Mtxt)
    M = LoadPickle("M.txt")
    dmin = 0.0055
    rhos = GetRho(M, dmin)
    deltas, minIndexs = GetDelta(M, rhos)
#    plt.scatter(rhos, deltas)

#    center = 0
#    index = []
#    i = 0
#    for delta, rho in zip(deltas, rhos):
#        if rho > 8 and delta > 0.015:
#            center += 1
#            index.append([i,tamples[i][0],rho,delta])
#        i+=1
#    print center ,index

    centerPoint = GetCenterPointSet(deltas, rhos)
    indexlist, labels = SetOtherPointLabel(minIndexs, tamples)

    errorNum = 0
    totalNum = 0
    for i, label in enumerate(labels):
        if label != tamples[i][0]:
            errorNum += 1
        totalNum += 1
    print 1.0*errorNum/totalNum, errorNum, totalNum
#    print centerPoint
#    np.savetxt('index.csv', np.transpose(indexlist), fmt='%d', delimiter=',')
