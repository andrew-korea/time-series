import numpy as np
import datetime as dt
import stumpy
import os

def MP(filename):
    serie = np.loadtxt('../data-sets/KDD-Cup/data/' + filename)#load the file
    bp = int(filename[16:-4]) #break point of training and testing set
    w = [20,50,100,300,600] #possible window size for matrix profiling
    best = 0
    diff = 1000000 #arbitrary large number
    train = 0
    #validation to select the best window size for the serie
    for i in w:
        temp = stumpy.stump(serie[0:bp], i)
        mx = np.sort(temp[:, 0])[-1]
        mean = np.mean(temp[:, 0])
        if (mx-mean) < diff:
            diff = mx - mean
            best = i
            train = temp
    mp = stumpy.stump(serie[bp:], best) #matrix profiling with the best window size
    outlier = np.argsort(mp[:, 0])[-1] 

    #confidence is measured by how much the max is above the mean and normalized
    #confidence = (max(mp[:, 0]) - np.mean(mp[:, 0]))/(max(mp[:, 0]) - min(mp[:, 0])) 

    lower = np.array(mp[:,0])[0:max(1,outlier - 600)]
    upper = np.array(mp[:,0])[min(len(mp)-2,outlier + 600):-1]
    confidence = (max(mp[:,0]) - max(max(lower),max(upper)))/max(mp[:,0])


    outlier = np.argsort(mp[:, 0])[-1] + bp #The max of the mp output is the anomaly

    return confidence, outlier

