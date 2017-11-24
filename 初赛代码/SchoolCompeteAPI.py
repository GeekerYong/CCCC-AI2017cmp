# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        dataSet.append(line)
    return dataSet
      
#将得到的数据变成list形式   
def normalize(dataSet, flag=0):
    resultData = []
    #对每一行数据做处理
    dataLen = len(dataSet)
    for i in range(0, dataLen):
        #训练和测试分开做处理
        if flag == 0: id, trail, dest, label = dataSet[i].strip().split(' ')
        else: 
            id, trail, dest = dataSet[i].strip().split(' ')       #对轨迹做处理
        resultTrail = []
        tempTrail = trail.split(';')
        TrailLen = len(tempTrail)
        for j in range(0, TrailLen):
            temp = tempTrail[j].split(',')
            if len(temp) == 3:
                resultTrail.append(temp)
        #对目的地做处理
        resultDest = dest.split(',')
        #整体加入结果中
        if flag == 0: resultData.append([id, resultTrail, resultDest, label])
        else: resultData.append([id, resultTrail, resultDest])
    return resultData

#计算速度的大小
def calSpeed(inA, inB):
    inA = np.mat(map(float, inA))
    inB = np.mat(map(float, inB))
    dist = np.linalg.norm(inB[0, :2] - inA[0, :2])
    if np.abs(inB[0, 2] - inA[0, 2]) < 1e-8: return 0
    else: delta = np.abs(inB[0, 2] - inA[0, 2])
    speed = dist / (delta)
    return speed

#计算x轴或者y轴的速度大小
def calXOrYSpeed(inA, inB, flag=0):
    if np.abs(inB[2] - inA[2]) < 1e-8: return 0
    else: delta = (inB[2] - inA[2])
    if flag == 0:
        offSet = inB[0] - inA[0]
    else:
        offSet = inB[1] - inA[1]
    return offSet / delta

#计算加速度大小
def calAcceleration(inA, inB):
    deltaSpeed = inB[0] - inA[0]
    if np.abs(inB[1] - inA[1]) < 1e-8: return 0
    else: deltaTime = np.abs(inB[1] - inA[1])
    return deltaSpeed / deltaTime
    
#计算时间差
def calDeltaTime(inA, inB):
    return np.abs(inB[2] - inA[2])
    
#计算偏移量
def caloffSet(inA, inB):
    inA = np.mat(map(float, inA))
    inB = np.mat(map(float, inB))
    dist = np.linalg.norm(inB[0, :2] - inA[0, :2])
    return dist

#计算x轴或者y轴坐标的偏移量
def calXOrYOffSet(inA, inB, flag=0):
    if flag == 0: return inB[0] - inA[0]
    else: return inB[1] - inA[1]

#计算加速度对应的时间
def calAccelTime(inA, inB):
    return (inB[2] + inA[2]) / 2

#计算非重复占比
def calCount(inA):
    temp = np.unique(inA)
    return (float)(len(temp)) / (float)(len(inA))

#计算角度    
def calAngle(inA, inB):
    deltaY = np.abs(inB[1] - inA[1])
    deltaX = np.abs(inB[0] - inA[0])
    if np.equal(deltaX, 0.0): return np.pi / 2 
    return np.arctan(deltaY / deltaX)
    
#计算角速度
def calAngleSpeed(inA, inB):
    if np.abs(inB[1] - inA[1]) < 1e-8: return 0
    else: delta = np.abs(inB[1] - inA[1])
    angleDelta = np.abs(inB[0] - inA[0])
    return angleDelta / delta
    
#计算回退次数
def calNum(seq, threshold):
    count = []
    for i in range(len(seq)):
        if seq[i] < threshold:
            count.append(seq[i])
    return len(count)

#计算众数
def calMode(seq):
    count_dict = {}
    for i in seq:  
        if count_dict.has_key(i):  
            count_dict[i] += 1
        else:  
            count_dict[i] = 1
            
    max_appear = 0
    for value in count_dict.values():  
        if value > max_appear:  
            max_appear = value           
    if max_appear == 1:  
        return np.mean(seq)
        
    mode_list=[]  
    for key, value in count_dict.items():  
        if value == max_appear:  
            mode_list.append(key)
    return np.mean(mode_list)
    
def calLenRatio(data1, data2):
    len1 = len(data1)
    len2 = len(data2)
    if len2 == 0:
        return 0.0
    else:
        return float(len1) / float(len2)
    
#添加速度以及与之相关的特征
def addSpeedAndSpeedFeat(dataNormlized):
    tempNormData = copy.deepcopy(dataNormlized)
    length = len(tempNormData)
    for i in range(0, length):
        #处理每条数据中的轨迹
        trailData = tempNormData[i][1]
        trailDataLen = len(trailData)
        
        for j in range(len(trailData)):
            trailData[j] = map(float, trailData[j])
            trailData[j] = map(abs, trailData[j])
        
        destData = tempNormData[i][2]
        destData = map(float, destData)
        destData = map(abs, destData)
        oneThirdThreshold = (trailData[trailDataLen-1][2] - trailData[0][2]) * 2 / 3
        oneFouthThreshold = (trailData[trailDataLen-1][2] - trailData[0][2]) * 3 / 4
        oneFifthThreshold = (trailData[trailDataLen-1][2] - trailData[0][2]) * 4 / 5
        
        #添加总速度，x轴速度,y轴速度
        if trailDataLen <= 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/3
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/4
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/5
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:        
            accelData = []
            speedResult = []
            oneThirdSpeedResult = []
            oneFouthSpeedResult = []
            oneFifthSpeedResult = []
            for j in range(0, trailDataLen - 1):
                #添加速度
                speed = calSpeed(trailData[j], trailData[j+1])
                speedResult.append(speed)
                #添加速度以及对应的时间
                speedTime = calAccelTime(trailData[j], trailData[j+1])
                accelData.append([speed, speedTime])
                #提取后1/3阶段的速度
                if trailData[j][2] >= oneThirdThreshold:
                    oneThirdSpeedResult.append(speed)
                #提取后1/4阶段的速度
                if trailData[j][2] >= oneFouthThreshold:
                    oneFouthSpeedResult.append(speed)
                #提取后1/5阶段的速度
                if trailData[j][2] >= oneFifthThreshold:
                    oneFifthSpeedResult.append(speed)
                    
            #速度的最大值、最小值、平均值、方差
            tempNormData[i].append(speedResult)
            tempNormData[i].append(np.max(speedResult))
            tempNormData[i].append(np.min(speedResult))
            tempNormData[i].append(np.mean(speedResult))
            tempNormData[i].append(np.var(speedResult))
            tempNormData[i].append(calCount(speedResult))
            tempNormData[i].append(calMode(speedResult))
            tempNormData[i].append(speedResult[-1])
            
            speedResultDiff = np.unique(speedResult)
            tempNormData[i].append(speedResultDiff)
            tempNormData[i].append(np.max(speedResultDiff))
            tempNormData[i].append(np.min(speedResultDiff))
            tempNormData[i].append(np.mean(speedResultDiff))
            tempNormData[i].append(np.var(speedResultDiff))
            tempNormData[i].append(len(speedResultDiff))
                
            #添加X轴速度
            xAccelData = []
            xSpeedResult = []
            xOneThirdSpeedResult = []
            xOneFouthSpeedResult = []
            xOneFifthSpeedResult = []         
            for j in range(0, trailDataLen - 1):
                #添加速度
                xSpeed = calXOrYSpeed(trailData[j], trailData[j+1])
                xSpeedResult.append(xSpeed)
                #添加速度以及对应的时间
                speedTime = calAccelTime(trailData[j], trailData[j+1])
                xAccelData.append([xSpeed, speedTime])                
                #提取后1/3阶段的速度
                if trailData[j][2] >= oneThirdThreshold:
                    xOneThirdSpeedResult.append(xSpeed)
                #提取后1/4阶段的速度
                if trailData[j][2] >= oneFouthThreshold:
                    xOneFouthSpeedResult.append(xSpeed)
                #提取后1/5阶段的速度
                if trailData[j][2] >= oneFifthThreshold:
                    xOneFifthSpeedResult.append(xSpeed)
                    
            tempNormData[i].append(xSpeedResult)
            tempNormData[i].append(np.max(xSpeedResult))
            tempNormData[i].append(np.min(xSpeedResult))
            tempNormData[i].append(np.mean(xSpeedResult))
            tempNormData[i].append(np.var(xSpeedResult))
            tempNormData[i].append(calCount(xSpeedResult))
            tempNormData[i].append(calMode(xSpeedResult))
            tempNormData[i].append(xSpeedResult[-1])
            
            xSpeedResultDiff = np.unique(xSpeedResult)
            tempNormData[i].append(xSpeedResultDiff)
            tempNormData[i].append(np.max(xSpeedResultDiff))
            tempNormData[i].append(np.min(xSpeedResultDiff))
            tempNormData[i].append(np.mean(xSpeedResultDiff))
            tempNormData[i].append(np.var(xSpeedResultDiff))
            tempNormData[i].append(len(xSpeedResultDiff))
                
            #添加y轴速度
            yAccelData = []
            ySpeedResult = []
            yOneThirdSpeedResult = []
            yOneFouthSpeedResult = []
            yOneFifthSpeedResult = []
            for j in range(0, trailDataLen - 1):
                #添加速度
                ySpeed = calXOrYSpeed(trailData[j], trailData[j+1], flag=1)
                ySpeedResult.append(ySpeed)
                #添加速度以及对应的时间
                speedTime = calAccelTime(trailData[j], trailData[j+1])
                yAccelData.append([ySpeed, speedTime])
                #提取后1/3阶段的速度
                if trailData[j][2] >= oneThirdThreshold:
                    yOneThirdSpeedResult.append(ySpeed)
                #提取后1/4阶段的速度
                if trailData[j][2] >= oneFouthThreshold:
                    yOneFouthSpeedResult.append(ySpeed)
                #提取后1/5阶段的速度
                if trailData[j][2] >= oneFifthThreshold:
                    yOneFifthSpeedResult.append(ySpeed)
                    
            tempNormData[i].append(ySpeedResult)
            tempNormData[i].append(np.max(ySpeedResult))
            tempNormData[i].append(np.mean(ySpeedResult))
            tempNormData[i].append(np.var(ySpeedResult))
            tempNormData[i].append(calCount(ySpeedResult))
            tempNormData[i].append(calMode(ySpeedResult))
            tempNormData[i].append(ySpeedResult[-1])
            
            ySpeedResultDiff = np.unique(ySpeedResult)
            tempNormData[i].append(ySpeedResultDiff)
            tempNormData[i].append(np.max(ySpeedResultDiff))
            tempNormData[i].append(np.min(ySpeedResultDiff))
            tempNormData[i].append(np.mean(ySpeedResultDiff))
            tempNormData[i].append(np.var(ySpeedResultDiff))
            tempNormData[i].append(len(ySpeedResultDiff))
            
            #添加后1/3时间段内速度特征
            if len(oneThirdSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneThirdSpeedResult)
                tempNormData[i].append(np.max(oneThirdSpeedResult))
                tempNormData[i].append(np.min(oneThirdSpeedResult))
                tempNormData[i].append(np.mean(oneThirdSpeedResult))
                tempNormData[i].append(np.var(oneThirdSpeedResult))
                tempNormData[i].append(len(oneThirdSpeedResult))
                tempNormData[i].append(calLenRatio(oneThirdSpeedResult, speedResult))  
            
            if len(xOneThirdSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneThirdSpeedResult)
                tempNormData[i].append(np.max(xOneThirdSpeedResult))
                tempNormData[i].append(np.min(xOneThirdSpeedResult))
                tempNormData[i].append(np.mean(xOneThirdSpeedResult))
                tempNormData[i].append(np.var(xOneThirdSpeedResult))
                tempNormData[i].append(len(xOneThirdSpeedResult)) 
                tempNormData[i].append(calLenRatio(xOneThirdSpeedResult, speedResult))  
            
            if len(yOneThirdSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneThirdSpeedResult)
                tempNormData[i].append(np.max(yOneThirdSpeedResult))
                tempNormData[i].append(np.min(yOneThirdSpeedResult))
                tempNormData[i].append(np.mean(yOneThirdSpeedResult))
                tempNormData[i].append(np.var(yOneThirdSpeedResult))
                tempNormData[i].append(len(yOneThirdSpeedResult))
                tempNormData[i].append(calLenRatio(yOneThirdSpeedResult, speedResult))  
                
            #添加后1/4时间段内速度特征
            if len(oneFouthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneFouthSpeedResult)
                tempNormData[i].append(np.max(oneFouthSpeedResult))
                tempNormData[i].append(np.min(oneFouthSpeedResult))
                tempNormData[i].append(np.mean(oneFouthSpeedResult))
                tempNormData[i].append(np.var(oneFouthSpeedResult))
                tempNormData[i].append(len(oneFouthSpeedResult))
                tempNormData[i].append(calLenRatio(oneFouthSpeedResult, speedResult))
            
            if len(xOneFouthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneFouthSpeedResult)
                tempNormData[i].append(np.max(xOneFouthSpeedResult))
                tempNormData[i].append(np.min(xOneFouthSpeedResult))
                tempNormData[i].append(np.mean(xOneFouthSpeedResult))
                tempNormData[i].append(np.var(xOneFouthSpeedResult))
                tempNormData[i].append(len(xOneFouthSpeedResult))
                tempNormData[i].append(calLenRatio(xOneFouthSpeedResult, speedResult))
            
            if len(yOneFouthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneFouthSpeedResult)
                tempNormData[i].append(np.max(yOneFouthSpeedResult))
                tempNormData[i].append(np.min(yOneFouthSpeedResult))
                tempNormData[i].append(np.mean(yOneFouthSpeedResult))
                tempNormData[i].append(np.var(yOneFouthSpeedResult))
                tempNormData[i].append(len(yOneFouthSpeedResult))
                tempNormData[i].append(calLenRatio(yOneFouthSpeedResult, speedResult))
                
            #添加后1/5时间段内的速度特征
            if len(oneFifthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneFifthSpeedResult)
                tempNormData[i].append(np.max(oneFifthSpeedResult))
                tempNormData[i].append(np.min(oneFifthSpeedResult))
                tempNormData[i].append(np.mean(oneFifthSpeedResult))
                tempNormData[i].append(np.var(oneFifthSpeedResult))
                tempNormData[i].append(len(oneFifthSpeedResult))
                tempNormData[i].append(calLenRatio(oneFifthSpeedResult, speedResult))
            
            if len(xOneFifthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneFifthSpeedResult)
                tempNormData[i].append(np.max(xOneFifthSpeedResult))
                tempNormData[i].append(np.min(xOneFifthSpeedResult))
                tempNormData[i].append(np.mean(xOneFifthSpeedResult))
                tempNormData[i].append(np.var(xOneFifthSpeedResult))
                tempNormData[i].append(len(xOneFifthSpeedResult)) 
                tempNormData[i].append(calLenRatio(xOneFifthSpeedResult, speedResult))
            
            if len(yOneFifthSpeedResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneFifthSpeedResult)
                tempNormData[i].append(np.max(yOneFifthSpeedResult))
                tempNormData[i].append(np.min(yOneFifthSpeedResult))
                tempNormData[i].append(np.mean(yOneFifthSpeedResult))
                tempNormData[i].append(np.var(yOneFifthSpeedResult))
                tempNormData[i].append(len(yOneFifthSpeedResult))
                tempNormData[i].append(calLenRatio(yOneFifthSpeedResult, speedResult))
                
        #添加加速度特征
        if trailDataLen <= 2:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)

            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/3
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/4
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            #1/5
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            #计算加速度
            accelResult = []
            oneThirdAccelResult = []
            oneFouthAccelResult = []
            oneFifthAccelResult = []
            accelDataLen = len(accelData)
            for j in range(0, accelDataLen - 1):
                acceleration = calAcceleration(accelData[j], accelData[j+1])
                accelResult.append(acceleration)             
                #提取后1/3阶段的加速度
                if accelData[j][1] >= oneThirdThreshold:
                    oneThirdAccelResult.append(acceleration)
                #提取后1/4阶段的加速度
                if accelData[j][1] >= oneFouthThreshold:
                    oneFouthAccelResult.append(acceleration)
                #提取后1/5阶段的加速度
                if accelData[j][1] >= oneFifthThreshold:
                    oneFifthAccelResult.append(acceleration)
                    
            #加速度的最大值、最小值、平均值、方差
            tempNormData[i].append(accelResult)
            tempNormData[i].append(np.max(accelResult))
            tempNormData[i].append(np.mean(accelResult))
            tempNormData[i].append(np.var(accelResult))
            tempNormData[i].append(calCount(accelResult))
            tempNormData[i].append(calMode(accelResult))
            tempNormData[i].append(accelResult[-1])
            
            accelResultDiff = np.unique(accelResult)
            tempNormData[i].append(accelResultDiff)
            tempNormData[i].append(np.max(accelResultDiff))
            tempNormData[i].append(np.min(accelResultDiff))
            tempNormData[i].append(np.mean(accelResultDiff))
            tempNormData[i].append(np.var(accelResultDiff))
            tempNormData[i].append(len(accelResultDiff))
            
            #计算x轴加速度
            xAccelResult = []
            xOneThirdAccelResult = []
            xOneFouthAccelResult = []
            xOneFifthAccelResult = []
            xAccelDataLen = len(xAccelData)
            for j in range(0, xAccelDataLen - 1):
                xAcceleration = calAcceleration(xAccelData[j], xAccelData[j+1])
                xAccelResult.append(xAcceleration)                
                #提取后1/3阶段的加速度
                if xAccelData[j][1] >= oneThirdThreshold:
                    xOneThirdAccelResult.append(xAcceleration)
                #提取后1/4阶段的加速度
                if xAccelData[j][1] >= oneFouthThreshold:
                    xOneFouthAccelResult.append(xAcceleration)
                #提取后1/5阶段的加速度
                if xAccelData[j][1] >= oneFifthThreshold:
                    xOneFifthAccelResult.append(xAcceleration)
            #加速度的最大值、最小值、平均值、方差
            tempNormData[i].append(xAccelResult)
            tempNormData[i].append(np.max(xAccelResult))
            tempNormData[i].append(np.mean(xAccelResult))
            tempNormData[i].append(np.var(xAccelResult))
            tempNormData[i].append(calCount(xAccelResult))
            tempNormData[i].append(calMode(xAccelResult))
            tempNormData[i].append(xAccelResult[-1])
            
            xAccelResultDiff = np.unique(xAccelResult)
            tempNormData[i].append(xAccelResultDiff)
            tempNormData[i].append(np.max(xAccelResultDiff))
            tempNormData[i].append(np.min(xAccelResultDiff))
            tempNormData[i].append(np.mean(xAccelResultDiff))
            tempNormData[i].append(np.var(xAccelResultDiff))
            tempNormData[i].append(len(xAccelResultDiff))
            
            #计算y轴加速度
            yAccelResult = []
            yOneThirdAccelResult = []
            yOneFouthAccelResult = []
            yOneFifthAccelResult = []
            yAccelDataLen = len(yAccelData)
            for j in range(0, yAccelDataLen - 1):
                yAcceleration = calAcceleration(yAccelData[j], yAccelData[j+1])
                yAccelResult.append(yAcceleration)
                #提取后1/3阶段的加速度
                if yAccelData[j][1] >= oneThirdThreshold:
                    yOneThirdAccelResult.append(yAcceleration)
                #提取后1/4阶段的加速度
                if yAccelData[j][1] >= oneFouthThreshold:
                    yOneFouthAccelResult.append(yAcceleration)
                #提取后1/5阶段的加速度
                if yAccelData[j][1] >= oneFifthThreshold:
                    yOneFifthAccelResult.append(yAcceleration)
            #加速度的最大值、最小值、平均值、方差
            tempNormData[i].append(yAccelResult)
            tempNormData[i].append(np.max(yAccelResult))
            tempNormData[i].append(np.mean(yAccelResult))
            tempNormData[i].append(np.var(yAccelResult))
            tempNormData[i].append(calCount(yAccelResult))
            tempNormData[i].append(calMode(yAccelResult))
            tempNormData[i].append(yAccelResult[-1])
            
            yAccelResultDiff = np.unique(yAccelResult)
            tempNormData[i].append(yAccelResultDiff)
            tempNormData[i].append(np.max(yAccelResultDiff))
            tempNormData[i].append(np.min(yAccelResultDiff))
            tempNormData[i].append(np.mean(yAccelResultDiff))
            tempNormData[i].append(np.var(yAccelResultDiff))
            tempNormData[i].append(len(yAccelResultDiff))
            
            #后1/3的加速度
            if len(oneThirdAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneThirdAccelResult)
                tempNormData[i].append(np.max(oneThirdAccelResult))
                tempNormData[i].append(np.min(oneThirdAccelResult))
                tempNormData[i].append(np.mean(oneThirdAccelResult))
                tempNormData[i].append(np.var(oneThirdAccelResult))
                tempNormData[i].append(len(np.unique(oneThirdAccelResult)))  
            
            if len(xOneThirdAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneThirdAccelResult)
                tempNormData[i].append(np.max(xOneThirdAccelResult))
                tempNormData[i].append(np.min(xOneThirdAccelResult))
                tempNormData[i].append(np.mean(xOneThirdAccelResult))
                tempNormData[i].append(np.var(xOneThirdAccelResult))
                tempNormData[i].append(len(np.unique(xOneThirdAccelResult)))
            
            if len(yOneThirdAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneThirdAccelResult)
                tempNormData[i].append(np.max(yOneThirdAccelResult))
                tempNormData[i].append(np.min(yOneThirdAccelResult))
                tempNormData[i].append(np.mean(yOneThirdAccelResult))
                tempNormData[i].append(np.var(yOneThirdAccelResult))
                tempNormData[i].append(len(np.unique(yOneThirdAccelResult)))
            
            #后1/4的加速度
            if len(oneFouthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneFouthAccelResult)
                tempNormData[i].append(np.max(oneFouthAccelResult))
                tempNormData[i].append(np.min(oneFouthAccelResult))
                tempNormData[i].append(np.mean(oneFouthAccelResult))
                tempNormData[i].append(np.var(oneFouthAccelResult))
                tempNormData[i].append(len(np.unique(oneFouthAccelResult)))  
            
            if len(xOneFouthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneFouthAccelResult)
                tempNormData[i].append(np.max(xOneFouthAccelResult))
                tempNormData[i].append(np.min(xOneFouthAccelResult))
                tempNormData[i].append(np.mean(xOneFouthAccelResult))
                tempNormData[i].append(np.var(xOneFouthAccelResult))
                tempNormData[i].append(len(np.unique(xOneFouthAccelResult)))
            
            if len(yOneFouthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneFouthAccelResult)
                tempNormData[i].append(np.max(yOneFouthAccelResult))
                tempNormData[i].append(np.min(yOneFouthAccelResult))
                tempNormData[i].append(np.mean(yOneFouthAccelResult))
                tempNormData[i].append(np.var(yOneFouthAccelResult))
                tempNormData[i].append(len(np.unique(yOneFouthAccelResult)))
                
            #后1/5的加速度
            if len(oneFifthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(oneFifthAccelResult)
                tempNormData[i].append(np.max(oneFifthAccelResult))
                tempNormData[i].append(np.min(oneFifthAccelResult))
                tempNormData[i].append(np.mean(oneFifthAccelResult))
                tempNormData[i].append(np.var(oneFifthAccelResult))
                tempNormData[i].append(len(np.unique(oneFifthAccelResult)))  
            
            if len(xOneFifthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(xOneFifthAccelResult)
                tempNormData[i].append(np.max(xOneFifthAccelResult))
                tempNormData[i].append(np.min(xOneFifthAccelResult))
                tempNormData[i].append(np.mean(xOneFifthAccelResult))
                tempNormData[i].append(np.var(xOneFifthAccelResult))
                tempNormData[i].append(len(np.unique(xOneFifthAccelResult)))
            
            if len(yOneFifthAccelResult) < 1:
                tempNormData[i].append([])
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
                tempNormData[i].append(-1)
            else:
                tempNormData[i].append(yOneFifthAccelResult)
                tempNormData[i].append(np.max(yOneFifthAccelResult))
                tempNormData[i].append(np.min(yOneFifthAccelResult))
                tempNormData[i].append(np.mean(yOneFifthAccelResult))
                tempNormData[i].append(np.var(yOneFifthAccelResult))
                tempNormData[i].append(len(np.unique(yOneFifthAccelResult)))
            
        #添加移动偏移量特征
        if trailDataLen <= 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            offSetResult = []
            for j in range(0, trailDataLen - 1):
                offSet = caloffSet(trailData[j], trailData[j+1])
                offSetResult.append(offSet)
            tempNormData[i].append(offSetResult)
            tempNormData[i].append(np.max(offSetResult))
            tempNormData[i].append(np.min(offSetResult))
            tempNormData[i].append(np.mean(offSetResult))
            tempNormData[i].append(np.var(offSetResult))
            tempNormData[i].append(calMode(offSetResult))
            tempNormData[i].append(caloffSet(trailData[trailDataLen-1], destData))
            
            #x轴偏移量
            xOffSetResult = []
            for j in range(0, trailDataLen - 1):
                xOffSet = calXOrYOffSet(trailData[j], trailData[j+1])
                xOffSetResult.append(xOffSet)
            tempNormData[i].append(xOffSetResult)
            tempNormData[i].append(np.max(xOffSetResult))
            tempNormData[i].append(np.min(xOffSetResult))
            tempNormData[i].append(np.mean(xOffSetResult))
            tempNormData[i].append(np.var(xOffSetResult))
            tempNormData[i].append(calNum(xOffSetResult, 0.0))
            tempNormData[i].append(calMode(xOffSetResult))
            tempNormData[i].append(calXOrYOffSet(trailData[trailDataLen-1], destData))

            #y轴偏移量
            yOffSetResult = []
            for j in range(0, trailDataLen - 1):
                yOffSet = calXOrYOffSet(trailData[j], trailData[j+1], flag=1)
                yOffSetResult.append(yOffSet)
            tempNormData[i].append(yOffSetResult)
            tempNormData[i].append(np.max(yOffSetResult))
            tempNormData[i].append(np.min(yOffSetResult))
            tempNormData[i].append(np.mean(yOffSetResult))
            tempNormData[i].append(np.var(yOffSetResult))
            tempNormData[i].append(calNum(yOffSetResult, 0.0))
            tempNormData[i].append(calMode(yOffSetResult))
            tempNormData[i].append(calXOrYOffSet(trailData[trailDataLen-1], destData, flag=1))
        
        #添加时间间隔特征
        if trailDataLen <= 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            deltaTimeResult= []
            for j in range(0, trailDataLen - 1):
                deltaTime = calDeltaTime(trailData[j], trailData[j+1])
                deltaTimeResult.append(deltaTime)
            tempNormData[i].append(deltaTimeResult)
            tempNormData[i].append(np.max(deltaTimeResult))
            tempNormData[i].append(np.min(deltaTimeResult))
            tempNormData[i].append(np.mean(deltaTimeResult))
            tempNormData[i].append(np.var(deltaTimeResult))
            tempNormData[i].append(calCount(deltaTimeResult))
            tempNormData[i].append(calMode(deltaTimeResult))
            tempNormData[i].append(calDeltaTime(trailData[0], trailData[-1]))
            
        #添加角度特征
        if trailDataLen <= 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            angleTemp = []
            angleResult = []
            for j in range(0, trailDataLen - 1):
                angle = calAngle(trailData[j], trailData[j+1])
                angleResult.append(angle)
                
                angleTime = calAccelTime(trailData[j], trailData[j+1])
                angleTemp.append([angle, angleTime])
            tempNormData[i].append(angleResult)
            tempNormData[i].append(np.max(angleResult))
            tempNormData[i].append(np.min(angleResult))
            tempNormData[i].append(np.mean(angleResult))
            tempNormData[i].append(np.var(angleResult))
            tempNormData[i].append(calMode(angleResult))
            tempNormData[i].append(calCount(angleResult))
            
            angleResultDiff = np.unique(angleResult)
            tempNormData[i].append(angleResultDiff)
            tempNormData[i].append(np.max(angleResultDiff))
            tempNormData[i].append(np.min(angleResultDiff))
            tempNormData[i].append(np.mean(angleResultDiff))
            tempNormData[i].append(np.var(angleResultDiff))
            tempNormData[i].append(len(angleResultDiff))
            
            angleResultNoneZero = [x for x in angleResult if x != 0]
            if len(angleResultNoneZero) == 0: angleResultNoneZero.append(0.0)
            tempNormData[i].append(angleResultNoneZero)
            tempNormData[i].append(np.max(angleResultNoneZero))
            tempNormData[i].append(np.min(angleResultNoneZero))
            tempNormData[i].append(np.mean(angleResultNoneZero))
            tempNormData[i].append(np.var(angleResultNoneZero))
            tempNormData[i].append(len(angleResultNoneZero))
            
        #添加角速度特征:
        if trailDataLen <= 2:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            angleSpeedResult = []
            angleTempLen = len(angleTemp)
            for j in range(0, angleTempLen - 1):
                angleSpeed = calAngleSpeed(angleTemp[j], angleTemp[j+1])
                angleSpeedResult.append(angleSpeed)
            tempNormData[i].append(angleSpeedResult)
            tempNormData[i].append(np.max(angleSpeedResult))
            tempNormData[i].append(np.min(angleSpeedResult))
            tempNormData[i].append(np.mean(angleSpeedResult))
            tempNormData[i].append(np.var(angleSpeedResult))
            tempNormData[i].append(calCount(angleSpeedResult))
            tempNormData[i].append(calMode(angleSpeedResult))
            
            angleSpeedResultDiff = np.unique(angleSpeedResult)
            tempNormData[i].append(angleSpeedResultDiff)
            tempNormData[i].append(np.max(angleSpeedResultDiff))
            tempNormData[i].append(np.min(angleSpeedResultDiff))
            tempNormData[i].append(np.mean(angleSpeedResultDiff))
            tempNormData[i].append(np.var(angleSpeedResultDiff))
            tempNormData[i].append(len(angleSpeedResultDiff))
            
            angleSpeedResultNoneZero = [x for x in angleSpeedResult if x != 0]
            if len(angleSpeedResultNoneZero) == 0: angleSpeedResultNoneZero.append(0.0)
            tempNormData[i].append(angleSpeedResultNoneZero)
            tempNormData[i].append(np.max(angleSpeedResultNoneZero))
            tempNormData[i].append(np.min(angleSpeedResultNoneZero))
            tempNormData[i].append(np.mean(angleSpeedResultNoneZero))
            tempNormData[i].append(np.var(angleSpeedResultNoneZero))
            tempNormData[i].append(len(angleSpeedResultNoneZero))
            
        #添加坐标轴的一些特征
        if trailDataLen < 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            xResult = []
            for j in range(0, trailDataLen):
                x = float(trailData[j][0])
                xResult.append(x)
            tempNormData[i].append(xResult)
            tempNormData[i].append(np.max(xResult))
            tempNormData[i].append(np.min(xResult))
            tempNormData[i].append(np.mean(xResult))
            tempNormData[i].append(np.var(xResult))
            tempNormData[i].append(calCount(xResult))
            tempNormData[i].append(calMode(xResult))
            
            yResult = []
            for j in range(0, trailDataLen):
                y = float(trailData[j][1])
                yResult.append(y)
            tempNormData[i].append(yResult)
            tempNormData[i].append(np.max(yResult))
            tempNormData[i].append(np.min(yResult))
            tempNormData[i].append(np.mean(yResult))
            tempNormData[i].append(np.var(yResult))
            tempNormData[i].append(calCount(yResult))
            tempNormData[i].append(calMode(yResult))
            
        #添加时间的一些特征：
        if trailDataLen < 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            timeResult = []
            for j in range(0, trailDataLen):
                time = float(trailData[j][2])
                timeResult.append(time)
            tempNormData[i].append(timeResult)
            tempNormData[i].append(np.max(timeResult))
            tempNormData[i].append(np.min(timeResult))
            tempNormData[i].append(np.mean(timeResult))
            tempNormData[i].append(np.var(timeResult))
            tempNormData[i].append(len(timeResult))
            
        #添加每个点到目的地的距离
        if trailDataLen < 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            xAimDistResult = []
            for j in range(0, trailDataLen):
                xAimDist = calXOrYOffSet(trailData[j], destData)
                xAimDistResult.append(xAimDist)
            tempNormData[i].append(xAimDistResult)
            tempNormData[i].append(np.max(xAimDistResult))
            tempNormData[i].append(np.min(xAimDistResult))
            tempNormData[i].append(np.mean(xAimDistResult))
            tempNormData[i].append(np.var(xAimDistResult))
            tempNormData[i].append(calMode(xAimDistResult))
            tempNormData[i].append(calCount(xAimDistResult))
            
            yAimDistResult = []
            for j in range(0, trailDataLen):
                yAimDist = calXOrYOffSet(trailData[j], destData, flag=1)
                yAimDistResult.append(yAimDist)
            tempNormData[i].append(yAimDistResult)
            tempNormData[i].append(np.max(yAimDistResult))
            tempNormData[i].append(np.min(yAimDistResult))
            tempNormData[i].append(np.mean(yAimDistResult))
            tempNormData[i].append(np.var(yAimDistResult))
            tempNormData[i].append(calMode(yAimDistResult))
            tempNormData[i].append(calCount(yAimDistResult))
            
        #添加到目的地的角度特征
        if trailDataLen < 1:
            tempNormData[i].append([])
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
            tempNormData[i].append(-1)
        else:
            aimAngleResult = []
            for j in range(0, trailDataLen):
                aimAngle = calAngle(trailData[j], destData)
                aimAngleResult.append(aimAngle)
                
            tempNormData[i].append(aimAngleResult)
            tempNormData[i].append(np.max(aimAngleResult))
            tempNormData[i].append(np.min(aimAngleResult))
            tempNormData[i].append(np.mean(aimAngleResult))
            tempNormData[i].append(np.var(aimAngleResult))
            tempNormData[i].append(calMode(aimAngleResult))
            tempNormData[i].append(calCount(aimAngleResult))
            
    return tempNormData

#绘图函数
def show_trace_pic(trace,index):
    point_list = []
    for point in trace:
        if(point[0]==''):
            break
        temp = []
        temp.append(point[0])
        temp.append(point[1])
        point_list.append(tuple(temp))
        
        plt.figure(figsize=(7, 5))
        plt.plot(point_list, linewidth=1)
        plt.legend()
        foo_fig = plt.gcf()
        foo_fig.savefig('test'+str(index)+'.png')
        plt.show()
        
#绘制散点图
def show_scatter_pic(trainData, column):
    train_0 = trainData[trainData.label == 0]
    train_1 = trainData[trainData.label == 1]   
    
    plt.figure()
    plt.scatter(train_0['id'].tolist(), train_0[column].tolist(), marker='x', color='m', s=30)
    plt.scatter(train_1['id'].tolist(), train_1[column].tolist(), marker='+', color='r', s=10)
    plt.xlabel('id')
    plt.ylabel(column)    
    plt.savefig(column + '.jpg')
    
#绘制速度等坐标图
def showResult(data1, data2):
    for i in range(len(data1)):
        plt.figure(figsize=(7, 5))
        plt.plot(data1[i][:-1], data2[i])
        plt.savefig('./ySpeed/ySpeed_'+ str(i) + '.png')
        plt.close('all')
    
# 写结果
def write_result(result):
    robot_list = []
    f = open("./output/BDC1445_20170722.txt",'w+')
    for i in range(len(result)):
        if result[i] == 0:
            print i
            robot_list.append(i+1)
            f.write(str(i+1))
            f.write("\n")
    f.close()
    return robot_list
    
def writeResult(data):
    f = open('./ySpeed.txt', "w+")
    for i in range(len(data)):
        f.write(str(data[i]))
        f.write('\n')
    f.close()