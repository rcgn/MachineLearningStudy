# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:55:16 2016

@author: zz
"""

import copy
import random as random
import numpy as np
import pickle


def LoadPickle():
    try:
        with open("train.txt", 'rb') as fTrain, open("test.txt", 'rb') as fTest:
            train = pickle.load(fTrain)
            test = pickle.load(fTest)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
    return train, test


def GetInitialTemplate(train):
    templates = dict()

    for a in train:
        if a[0] not in templates.keys():
            templates[a[0]] = a
    return templates


def GetTampleLabel(template, tample):
    resultLabel = None
    minDistance = float('inf')
    for label in template.keys():
        distance = np.sum(np.power(tample[1:]-template[label][1:], 2))
        if distance < minDistance:
            minDistance = distance
            resultLabel = label
    return resultLabel


def GetNewTemplate(templates, tample, mu):
    label = GetTampleLabel(templates, tample)
    if(label == tample[0]):
        templates[label] = templates[label] + mu * (tample-templates[label])
        templates[label][0] = label
    else:
        templates[label] = templates[label] - mu * (tample-templates[label])
        templates[label][0] = label
    return templates


def SaveTemplates(templates, name):
    templatesList = None
    for label in templates:
        if templatesList is None:
            templatesList = copy.copy(templates[label])
        else:
            templatesList = np.vstack((templatesList, templates[label]))
    np.savetxt(name, templatesList, delimiter=',', fmt='%f')
    print templatesList


def LVQTrain(train, mu, cycleIndex):
    templates = GetInitialTemplate(train)
#    SaveTemplates(templates, "templates1.csv")
    for _ in range(cycleIndex):
        i = random.randint(0, train.shape[0]-1)
        tample = train[i]
        templates = GetNewTemplate(templates, tample, mu)
#    SaveTemplates(templates, "templates2.csv")
    for tample in train:
        templates = GetNewTemplate(templates, tample, mu)
#    SaveTemplates(templates, "templates3.csv")
    return templates


def LVQTest(tample, templates):
    return GetTampleLabel(templates, tample)


if __name__ == "__main__":
    train, test = LoadPickle()
    mu = 0.001
    cycleIndex = 5000
    templates = LVQTrain(train, mu, cycleIndex)
    totalNum = 0
    errorNum = 0
    errorRate = 0.0
    for tample in train:
        label = LVQTest(tample, templates)
        if label != tample[0]:
            errorNum += 1
        totalNum += 1
        print label, tample[0], label == tample[0]
    print errorNum, totalNum, 1.0*errorNum/totalNum
