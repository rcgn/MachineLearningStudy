# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 15:43:16 2016

@author: zz
"""
import math
from matplotlib import pyplot as plt

#计算损失函数导数
def GetGradient(x):
    return 10*math.pow(x,9)

#计算损失函数值
def GetF_x(x):
    return math.pow(x,10)

#计算以学习率为自变量的损失函数值
def GetH_alpha(alpha,dif,x):
    return -GetGradient(x-alpha*dif)*dif
    
#计算二分法端点
def GetAlphaSearchEnd(x):
    alpha = 1
    
    dif = GetGradient(x)
    for _ in range(50):   
        hGradient = GetH_alpha(alpha,dif,x)
        if hGradient > 0:
            break
        else:
            alpha *= 2
    return alpha

#二分法搜索学习率值    
def BisectionLineSearch(x):
    start = 0
    end=GetAlphaSearchEnd(x)
    
    dif = GetGradient(x)

    while end - start > 0.001:
        mid = 0.5*(start+end)
        hMid = GetH_alpha(mid,dif,x)
        if hMid<=0:
            start = mid
        else:
            end = mid

    return 0.5*(start + end)
       
#线性回溯搜索alpha值    
def BackingLineSearch(x):#Backing Line Search
    c = 0.3  
    end = 1.0
    dif = GetGradient(x)   
    f_x = GetF_x(x)
    for _ in range(30):
        f_end = GetF_x(x-end*dif)
        if f_end > f_x:
            break
        else:
            end *= 2
    
    alpha = end
    for _ in range(50):
        f_alpha = GetF_x(x-alpha*dif)
        if f_alpha - f_x < -c*alpha*dif*dif:
            break
        else:
            alpha /= 2.0

    return alpha

#学习率计算函数    
def GetAlpha(x,flag):
    if flag == 1:
        return 0.01  
    elif flag == 2:
        return BisectionLineSearch(x)
    elif flag == 3:
        return BackingLineSearch(x)

#X值迭代
def GetNextX(x,flag):
    alpha = GetAlpha(x,flag)
    gradient = GetGradient(x)
    return x-alpha*gradient
    

if __name__ == '__main__':
    flag = 1
    x = 1.5    
    threshold = 0.003
    n = 0
    err = GetF_x(x)
    errs = [err]
    xList = [x]
    while abs(x) > threshold and n<5000:
        x = GetNextX(x,flag)
        n += 1
        err = GetF_x(x)
        errs.append(err)
        xList.append(x)
        print n,err,x
#    plt.plot(errs)
    plt.plot(xList)