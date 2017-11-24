# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:02:14 2017

@author: sw
"""

import sys
from operator import add
import numpy as np
from pyspark import SparkContext
import os

###########################################################################
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

#########################################################################



def normalize(data, flag=0):
    resultData = []
    # 对训练集进行处理
    if flag==0:
        id, trail, dest, label = data.strip().split(' ')
    else:
        id, trail, dest = data.strip().split(' ')
    resultTrail = []
    tempTrail = trail.split(';')
    TrailLen = len(tempTrail)
    for i in range(0, TrailLen):
        temp = tempTrail[i].split(',')
        if len(temp) == 3:
            resultTrail.append(temp)
    resultDest = dest.split(',')
    if flag == 0:
        resultData.append(id)
        resultData.append(resultTrail)
        resultData.append(resultDest)
        resultData.append(label)
    else: 
        resultData.append(id)
        resultData.append(resultTrail)
        resultData.append(resultDest)
    return resultData
#
def generate_features(data, flag = 0, clf = 'dense'):
    tempNormData = []
    
    trailData = data[1]
    trailDataLen = len(trailData)
    
    for j in range(len(trailData)):
        trailData[j] = map(float, trailData[j])
        trailData[j] = map(abs, trailData[j])
        
    destData = data[2]
    destData = map(float, destData)
    destData = map(abs, destData)
    oneThirdThreshold = (trailData[trailDataLen-1][2] - trailData[0][2]) * 2 / 3

    #添加总速度，x轴速度,y轴速度
    if trailDataLen <= 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
            
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:        
        accelData = []
        speedResult = []
        oneThirdSpeedResult = []
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
                
        #速度的最大值、最小值、平均值、方差
        tempNormData.append(np.max(speedResult))
        tempNormData.append(np.min(speedResult))
        tempNormData.append(np.mean(speedResult))
        tempNormData.append(np.var(speedResult))
        tempNormData.append(calCount(speedResult))
        tempNormData.append(calMode(speedResult))
        tempNormData.append(speedResult[-1])
        
        speedResultDiff = np.unique(speedResult)
        tempNormData.append(np.max(speedResultDiff))
        tempNormData.append(np.min(speedResultDiff))
        tempNormData.append(np.mean(speedResultDiff))
        tempNormData.append(np.var(speedResultDiff))
        tempNormData.append(len(speedResultDiff))
		
        #添加X轴速度
        xAccelData = []
        xSpeedResult = []
        xOneThirdSpeedResult = []
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
                
        tempNormData.append(np.max(xSpeedResult))
        tempNormData.append(np.min(xSpeedResult))
        tempNormData.append(np.mean(xSpeedResult))
        tempNormData.append(np.var(xSpeedResult))
        tempNormData.append(calCount(xSpeedResult))
        tempNormData.append(calMode(xSpeedResult))
        tempNormData.append(xSpeedResult[-1])
		
        xSpeedResultDiff = np.unique(xSpeedResult)
        tempNormData.append(np.max(xSpeedResultDiff))
        tempNormData.append(np.min(xSpeedResultDiff))
        tempNormData.append(np.mean(xSpeedResultDiff))
        tempNormData.append(np.var(xSpeedResultDiff))
        tempNormData.append(len(xSpeedResultDiff))
        
        #添加y轴速度
        yAccelData = []
        ySpeedResult = []
        yOneThirdSpeedResult = []
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
            
        tempNormData.append(np.max(ySpeedResult))
        tempNormData.append(np.mean(ySpeedResult))
        tempNormData.append(np.var(ySpeedResult))
        tempNormData.append(calCount(ySpeedResult))
        tempNormData.append(calMode(ySpeedResult))
        tempNormData.append(ySpeedResult[-1])
        
        ySpeedResultDiff = np.unique(ySpeedResult)
        tempNormData.append(np.max(ySpeedResultDiff))
        tempNormData.append(np.min(ySpeedResultDiff))
        tempNormData.append(np.mean(ySpeedResultDiff))
        tempNormData.append(np.var(ySpeedResultDiff))
        tempNormData.append(len(ySpeedResultDiff))
        
        #添加后1/3的速度特征
        if len(oneThirdSpeedResult) < 1:
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
        else:
            tempNormData.append(np.max(oneThirdSpeedResult))
            tempNormData.append(np.min(oneThirdSpeedResult))
            tempNormData.append(np.mean(oneThirdSpeedResult))
            tempNormData.append(np.var(oneThirdSpeedResult))
            tempNormData.append(len(oneThirdSpeedResult))
        
        if len(xOneThirdSpeedResult) < 1:
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
        else:
            tempNormData.append(np.max(xOneThirdSpeedResult))
            tempNormData.append(np.min(xOneThirdSpeedResult))
            tempNormData.append(np.mean(xOneThirdSpeedResult))
            tempNormData.append(np.var(xOneThirdSpeedResult))
            tempNormData.append(len(xOneThirdSpeedResult))
        
        if len(yOneThirdSpeedResult) < 1:
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
            tempNormData.append(-1)
        else:
            tempNormData.append(np.max(yOneThirdSpeedResult))
            tempNormData.append(np.min(yOneThirdSpeedResult))
            tempNormData.append(np.mean(yOneThirdSpeedResult))
            tempNormData.append(np.var(yOneThirdSpeedResult))
            tempNormData.append(len(yOneThirdSpeedResult))
        
    #添加加速度特征
    if trailDataLen <= 2:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        #计算加速度
        accelResult = []
        accelDataLen = len(accelData)
        for j in range(0, accelDataLen - 1):
            acceleration = calAcceleration(accelData[j], accelData[j+1])
            accelResult.append(acceleration)
        #加速度的最大值、最小值、平均值、方差
        #tempNormData.append(accelResult)
        tempNormData.append(np.max(accelResult))
        tempNormData.append(np.mean(accelResult))
        tempNormData.append(np.var(accelResult))
        tempNormData.append(calCount(accelResult))
        tempNormData.append(calMode(accelResult))
        tempNormData.append(accelResult[-1])
        
        accelResultDiff = np.unique(accelResult)
        tempNormData.append(np.max(accelResultDiff))
        tempNormData.append(np.min(accelResultDiff))
        tempNormData.append(np.mean(accelResultDiff))
        tempNormData.append(np.var(accelResultDiff))
        tempNormData.append(len(accelResultDiff))
        
        #计算x轴加速度
        xAccelResult = []
        xAccelDataLen = len(xAccelData)
        for j in range(0, xAccelDataLen - 1):
            xAcceleration = calAcceleration(xAccelData[j], xAccelData[j+1])
            xAccelResult.append(xAcceleration)
        #加速度的最大值、最小值、平均值、方差
        tempNormData.append(np.max(xAccelResult))
        tempNormData.append(np.mean(xAccelResult))
        tempNormData.append(np.var(xAccelResult))
        tempNormData.append(calCount(xAccelResult))
        tempNormData.append(calMode(xAccelResult))
        tempNormData.append(xAccelResult[-1])
        
        xAccelResultDiff = np.unique(xAccelResult)
        tempNormData.append(np.max(xAccelResultDiff))
        tempNormData.append(np.min(xAccelResultDiff))
        tempNormData.append(np.mean(xAccelResultDiff))
        tempNormData.append(np.var(xAccelResultDiff))
        tempNormData.append(len(xAccelResultDiff))
		
        #计算y轴加速度
        yAccelResult = []
        yAccelDataLen = len(yAccelData)
        for j in range(0, yAccelDataLen - 1):
            yAcceleration = calAcceleration(yAccelData[j], yAccelData[j+1])
            yAccelResult.append(yAcceleration)
        #加速度的最大值、最小值、平均值、方差
        tempNormData.append(np.max(yAccelResult))
        tempNormData.append(np.mean(yAccelResult))
        tempNormData.append(np.var(yAccelResult))
        tempNormData.append(calCount(yAccelResult))
        tempNormData.append(calMode(yAccelResult))
        tempNormData.append(yAccelResult[-1])
        
        yAccelResultDiff = np.unique(yAccelResult)
        tempNormData.append(np.max(yAccelResultDiff))
        tempNormData.append(np.min(yAccelResultDiff))
        tempNormData.append(np.mean(yAccelResultDiff))
        tempNormData.append(np.var(yAccelResultDiff))
        tempNormData.append(len(yAccelResultDiff))
        
    #添加移动偏移量特征
    if trailDataLen <= 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        offSetResult = []
        for j in range(0, trailDataLen - 1):
            offSet = caloffSet(trailData[j], trailData[j+1])
            offSetResult.append(offSet)
        tempNormData.append(np.max(offSetResult))
        tempNormData.append(np.min(offSetResult))
        tempNormData.append(np.mean(offSetResult))
        tempNormData.append(np.var(offSetResult))
        tempNormData.append(calMode(offSetResult))
        tempNormData.append(caloffSet(trailData[trailDataLen-1], destData))
        
        #x轴偏移量
        xOffSetResult = []
        for j in range(0, trailDataLen - 1):
            xOffSet = calXOrYOffSet(trailData[j], trailData[j+1])
            xOffSetResult.append(xOffSet)
        #tempNormData.append(xOffSetResult)
        tempNormData.append(np.max(xOffSetResult))
        tempNormData.append(np.min(xOffSetResult))
        tempNormData.append(np.mean(xOffSetResult))
        tempNormData.append(np.var(xOffSetResult))
        tempNormData.append(calNum(xOffSetResult, 0.0))
        tempNormData.append(calMode(xOffSetResult))
        tempNormData.append(calXOrYOffSet(trailData[trailDataLen-1], destData))
        
        #y轴偏移量
        yOffSetResult = []
        for j in range(0, trailDataLen - 1):
            yOffSet = calXOrYOffSet(trailData[j], trailData[j+1], flag=1)
            yOffSetResult.append(yOffSet)
            
        tempNormData.append(np.max(yOffSetResult))
        tempNormData.append(np.min(yOffSetResult))
        tempNormData.append(np.mean(yOffSetResult))
        tempNormData.append(np.var(yOffSetResult))
        tempNormData.append(calNum(yOffSetResult, 0.0))
        tempNormData.append(calMode(yOffSetResult))
        tempNormData.append(calXOrYOffSet(trailData[trailDataLen-1], destData, flag=1))
    
    #添加时间间隔特征
    if trailDataLen <= 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        deltaTimeResult= []
        for j in range(0, trailDataLen - 1):
            deltaTime = calDeltaTime(trailData[j], trailData[j+1])
            deltaTimeResult.append(deltaTime)
        tempNormData.append(np.max(deltaTimeResult))
        tempNormData.append(np.min(deltaTimeResult))
        tempNormData.append(np.mean(deltaTimeResult))
        tempNormData.append(np.var(deltaTimeResult))
        tempNormData.append(calCount(deltaTimeResult))
        tempNormData.append(calMode(deltaTimeResult))
        tempNormData.append(calDeltaTime(trailData[0], trailData[-1]))
        
    #添加角度特征
    if trailDataLen <= 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        angleTemp = []
        angleResult = []
        for j in range(0, trailDataLen - 1):
            angle = calAngle(trailData[j], trailData[j+1])
            angleResult.append(angle)
            
            angleTime = calAccelTime(trailData[j], trailData[j+1])
            angleTemp.append([angle, angleTime])
			
        tempNormData.append(np.max(angleResult))
        tempNormData.append(np.min(angleResult))
        tempNormData.append(np.mean(angleResult))
        tempNormData.append(np.var(angleResult))
        tempNormData.append(calMode(angleResult))
        tempNormData.append(calCount(angleResult))
        
        angleResultDiff = np.unique(angleResult)
        tempNormData.append(np.max(angleResultDiff))
        tempNormData.append(np.min(angleResultDiff))
        tempNormData.append(np.mean(angleResultDiff))
        tempNormData.append(np.var(angleResultDiff))
        tempNormData.append(len(angleResultDiff))
        
        angleResultNoneZero = [x for x in angleResult if x != 0]
        if len(angleResultNoneZero) == 0: angleResultNoneZero.append(0.0)
        tempNormData.append(np.max(angleResultNoneZero))
        tempNormData.append(np.min(angleResultNoneZero))
        tempNormData.append(np.mean(angleResultNoneZero))
        tempNormData.append(np.var(angleResultNoneZero))
        tempNormData.append(len(angleResultNoneZero))
        
    #添加角速度特征:
    if trailDataLen <= 2:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        angleSpeedResult = []
        angleTempLen = len(angleTemp)
        for j in range(0, angleTempLen - 1):
            angleSpeed = calAngleSpeed(angleTemp[j], angleTemp[j+1])
            angleSpeedResult.append(angleSpeed)

        tempNormData.append(np.max(angleSpeedResult))
        tempNormData.append(np.min(angleSpeedResult))
        tempNormData.append(np.mean(angleSpeedResult))
        tempNormData.append(np.var(angleSpeedResult))
        tempNormData.append(calCount(angleSpeedResult))
        tempNormData.append(calMode(angleSpeedResult))
        
        angleSpeedResultDiff = np.unique(angleSpeedResult)
        tempNormData.append(np.max(angleSpeedResultDiff))
        tempNormData.append(np.min(angleSpeedResultDiff))
        tempNormData.append(np.mean(angleSpeedResultDiff))
        tempNormData.append(np.var(angleSpeedResultDiff))
        tempNormData.append(len(angleSpeedResultDiff))
        
    #添加坐标轴的一些特征
    if trailDataLen < 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        xResult = []
        for j in range(0, trailDataLen):
            x = float(trailData[j][0])
            xResult.append(x)
        tempNormData.append(np.max(xResult))
        tempNormData.append(np.min(xResult))
        tempNormData.append(np.mean(xResult))
        tempNormData.append(np.var(xResult))
        tempNormData.append(calCount(xResult))
        tempNormData.append(calMode(xResult))
        
        yResult = []
        for j in range(0, trailDataLen):
            y = float(trailData[j][1])
            yResult.append(y)
        tempNormData.append(np.max(yResult))
        tempNormData.append(np.min(yResult))
        tempNormData.append(np.mean(yResult))
        tempNormData.append(np.var(yResult))
        tempNormData.append(calCount(yResult))
        tempNormData.append(calMode(yResult))
    
    #添加时间的一些特征：
    if trailDataLen < 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        timeResult = []
        for j in range(0, trailDataLen):
            time = float(trailData[j][2])
            timeResult.append(time)
        
        tempNormData.append(np.max(timeResult))
        tempNormData.append(np.min(timeResult))
        tempNormData.append(np.mean(timeResult))
        tempNormData.append(np.var(timeResult))
        tempNormData.append(len(timeResult))
        
    #添加和目的地有关的特征
    if trailDataLen < 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)

        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        xAimDistResult = []
        for j in range(0, trailDataLen):
            xAimDist = calXOrYOffSet(trailData[j], destData)
            xAimDistResult.append(xAimDist)
			
        tempNormData.append(np.max(xAimDistResult))
        tempNormData.append(np.min(xAimDistResult))
        tempNormData.append(np.mean(xAimDistResult))
        tempNormData.append(np.var(xAimDistResult))
        tempNormData.append(calMode(xAimDistResult))
        tempNormData.append(calCount(xAimDistResult))
        
        yAimDistResult = []
        for j in range(0, trailDataLen):
            yAimDist = calXOrYOffSet(trailData[j], destData, flag=1)
            yAimDistResult.append(yAimDist)

        tempNormData.append(np.max(yAimDistResult))
        tempNormData.append(np.min(yAimDistResult))
        tempNormData.append(np.mean(yAimDistResult))
        tempNormData.append(np.var(yAimDistResult))
        tempNormData.append(calMode(yAimDistResult))
        tempNormData.append(calCount(yAimDistResult))
        
    if trailDataLen < 1:
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
        tempNormData.append(-1)
    else:
        aimAngleResult = []
        for j in range(0, trailDataLen):
            aimAngle = calAngle(trailData[j], destData)
            aimAngleResult.append(aimAngle)
            
        tempNormData.append(np.max(aimAngleResult))
        tempNormData.append(np.min(aimAngleResult))
        tempNormData.append(np.mean(aimAngleResult))
        tempNormData.append(np.var(aimAngleResult))
        tempNormData.append(calMode(aimAngleResult))
        tempNormData.append(calCount(aimAngleResult))

    if clf == 'dense':
        if flag==0:
            label = int(data[3])
            tempNormData.append(label)
        #dense
        fea_str_list = [str(item) for item in tempNormData]
        fea_str = ' '.join(fea_str_list)        
    else:
        #libsvm
        fea_str_list = ['{}:{}'.format(i+1,j ) for i,j in enumerate(tempNormData)]
        if flag ==0:
            fea_str_list = [str(data[3])] + fea_str_list
        else:
            fea_str_list = [str(data[0])] + fea_str_list
        fea_str = ' '.join(fea_str_list)
#        fea_str_list = []
#        if flag==0:
#            fea_str_list.append(str(data[3]))
#        cnt = 1
#        for fea in tempNormData:
#            fea_str_list.append(str(cnt)+":"+str(fea))
#            cnt = cnt +1
#        fea_str = ' '.join(fea_str_list)
    return fea_str
