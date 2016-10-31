# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:59:02 2016

@author: zz
"""

import pickle


def LoadPickle(filename):
    try:
        with open(filename, 'rb') as fTample:
            tamples = pickle.load(fTample)
    except:
        print "OPEN TRAIN OR TEST FILE ERROR!"
    return tamples


def WriteSvmFile(trainTamples, trainSvmFile):
    try:
        with open(trainSvmFile, 'w') as fTrain:
            for trainTample in trainTamples:
#                tmin = min(trainTample[1:])
#                tmax = max(trainTample[1:])
#                t = tmax - tmin 
                for i, num in enumerate(trainTample):
                    if i == 0:
                        line = str(int(trainTample[0]))
                    else:
#                        num = 2*num/t-1.0 #OK
#                        num = num-t/2.0-tmin #NOTOK
#                        num = 100*num #OK
                        line = line+' '+str(i)+':'+str(num)
                line = line+'\n'
                fTrain.write(line)
    except:
        print "OPEN SVM TRAIN FILE ERROR !"

if __name__ == "__main__":
#    trainTamples = LoadPickle('train.txt')
#    WriteSvmFile(trainTamples, "svmTrain.txt")

#    predictTamples = LoadPickle('tample.txt')
#    WriteSvmFile(predictTamples, "svmTample.txt")

    import os
    os.chdir('C:\Anaconda\libsvm-3.21\python')
    from svmutil import *
#    y, x = svm_read_problem('E:\Python\Project\SVM\svmTrain.txt')
#    m = svm_train(y, x,'-c 8 -g 100')
#    svm_save_model("train.model",m)

    m = svm_load_model("train.model")
    yt, xt = svm_read_problem('E:\Python\Project\SVM\svmTample.txt')
    p_label, p_acc, p_val = svm_predict(yt, xt, m)
