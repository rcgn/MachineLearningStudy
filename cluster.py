# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:20:29 2016

@author: zz
"""
import copy
import random as random
import numpy as np
import pickle

def LoadPickle():        
############保存测试和训练样本#########################
    try:
        with open("train.txt",'rb') as fTrain, open("test.txt",'rb') as fTest:
            train = pickle.load(fTrain)
            test = pickle.load(fTest)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
    return train,test
#############聚类####################################

    
def GetTampleLabel(template, tample):
    resultLabel = None
    minDistance = float('inf')
    for label in template.keys():
        distance = np.sum(np.power(tample[1:]-template[label][1:],2))
        if distance < minDistance:
            minDistance = distance
            resultLabel = label
            
    return resultLabel

def GetInitialTemplate_Random(train):
    template = dict()
    temp =set()
    i = 0
    while i<20:
        a = random.randint(0,train.shape[0]-1)
        if a not in temp:
            template[i] = train[a][:]
            temp.add(a)
            i+= 1
    return template

def GetInitialTemplate(train):
    template = dict()
    
    for a in train:
        if a[0] not in template.keys():
            template[a[0]] = a
    return template

def ClusterTrain(train):
    template = GetInitialTemplate(train)
    for _ in range(10):
        newTemplate = dict()    
        newTemplateNum = dict()
        
        for tample in train:
            label = GetTampleLabel(template,tample)
            if label not in newTemplate.keys():
                newTemplateNum[label] = 1
                newTemplate[label] = tample
            else:
                newTemplateNum[label] += 1
                newTemplate[label] =  newTemplate[label] + tample
        
        for label in newTemplate.keys():
            newTemplate[label] = newTemplate[label]/newTemplateNum[label]
        template = copy.copy(newTemplate)    
    return template
        
def ClusterTest(test,template):
    result =np.zeros((20,20))
    for tample in test:
        label = GetTampleLabel(template, tample)
        print label, tample[0]
        result[int(tample[0])][int(label)] += 1
    return result
     
train,test = LoadPickle()
     
template = ClusterTrain(train)
result = ClusterTest(train,template)

np.savetxt("result.csv",result,delimiter=',' ,fmt = '%d')