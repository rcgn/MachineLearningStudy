# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:15:12 2015

@author: zzheng
"""

import numpy as np
from matplotlib import pyplot as plt
import random as random
import pickle
# read currnecy text
message_filename = 'TWD1.tra'
messagefile = open(message_filename) 

messagefile.readline()
messagefile.readline()

featurenum = int(messagefile.readline())
valuenum = int(messagefile.readline())
values = messagefile.readline().split(' ')[:-1]
currency = messagefile.readline()


# read info text
infoname = 'TWD.tra'

infonum = 4+valuenum+featurenum
infotuple = tuple(range(infonum))
A = np.loadtxt(infoname, delimiter=' ',usecols = infotuple,unpack=False)


#整理分类数据
labels = []
template = dict()

for line in A:
    for i in range(4):
        if line[i] == 1.0:
            break
    index = i
    for i in range(4,4+valuenum):
        if line[i] == 1.0:
            break
    index = (i-4)*4+index
    labels.append(index)
    
tamples = A[:,4+valuenum:]
labels = np.array(labels).reshape(tamples.shape[0],1)
tamples = np.hstack((labels,tamples))

############将样本分为测试和训练两部分#########################
train, test = None, None
for tample in tamples:
    if(random.randint(0,1) is 0):
        if train is None:
            train = tample
        else:
            train = np.vstack((train,tample))
    else:
        if test is None:
            test = tample
        else:
            test = np.vstack((test,tample))
        
############保存测试和训练样本#########################
try:
    with open("train.txt",'wb') as fTrain, open("test.txt",'wb') as fTest, open("tample.txt",'wb') as fTample :
        pickle.dump(train,fTrain)
        pickle.dump(test,fTest)
        pickle.dump(tamples,fTample)
except:
    print "OPEN TRAIN OR TEST FILE ERROR!"

##############聚类####################################
#
#
#    
#def GetTampleLabel(template, tample):
#    resultLabel = None
#    minDistance = float('inf')
#    for label in template.keys():
#        distance = np.sum(np.power(tample[1:]-template[label],2))
#        if distance < minDistance:
#            minDistance = distance
#            resultLabel = label
#            
#    return resultLabel
#
#def GetInitialTemplate(train):
#    template = dict()
#    
#    for a in train:
#        if a[0] not in template.keys():
#            template[a[0]] = a
#    return template
#
#
#template = GetInitialTemplate(train)
#newTemplate = dict()    
#newTemplateNum = dict()
#
#for tample in train:
#    label = GetTampleLabel(template,tample)
#    if label not in newTemplate.keys():
#        newTemplateNum[label] = 1
#        newTemplate[label] = tample
#        
#
#      