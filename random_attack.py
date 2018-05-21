# -*- coding: UTF-8 -*-

import DNN
import random
import numpy as np
import json


def change_value(value):   #改变对应维的数据
    if value == 1:
        return 0
    if value == 0:
        return 1
    return value+int(random.uniform(0,1)*value)

def change_single_data(data):       #随机选择四个维并改变
    newdata=data.copy()
    col1 = random.randint(0, 4999)
    col2 = random.randint(0, 4999)
    col3 = random.randint(0, 4999)
    col4 = random.randint(0, 4999)
    newdata[col1] = change_value(newdata[col1])
    newdata[col2] = change_value(newdata[col2])
    newdata[col3] = change_value(newdata[col3])
    newdata[col4] = change_value(newdata[col4])
    return [col1,col2,col3,col4], newdata


def attack_single_data(data):       #对某一数据进行随机攻击，直到结果翻转，返回路径
    path=[]
    j=0
    while True:
        m=0
        change=[]
        baseline=DNN.get_benign_prob([data])[1]
        if baseline>0.5 and path is not None:
            print "end iter"
            return path
        next_data=[]
        for i in range(100):
            r,new_data=change_single_data(data)
            diff=DNN.get_benign_prob([new_data])[1]-baseline
            if diff>m:
                m=diff
                change=r
                next_data=new_data
                path.append([change,m])
        print str(j)+"th iter, "+"baseline: "+str(baseline)+" max diff:"+str(m)+" baseline:"+str(baseline)
        j+=1
        if len(next_data) !=0:
            data=next_data
    return path


# d=np.load('mdata_10000.npy')
# print d.shape
#
# p=[]
# for i in range(d.shape[0]):
#    p.append(attack_single_data(d[i]))
# p=np.array(p)
# np.save('single_10000.npy',p)
#
# f=data_process.input_data()
# DNN.test_given_data((f.test_data,f.test_label))






