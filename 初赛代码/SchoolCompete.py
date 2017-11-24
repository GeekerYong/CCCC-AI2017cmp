# -*- coding: utf-8 -*-

from SchoolCompeteAPI import *
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
from sklearn import cross_validation as cv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb

# 读取数据并对数据进行标准化处理
trainDataSet = loadDataSet('./data/dsjtzs_txfz_training.txt')
normlizedData_train = normalize(trainDataSet,0)

testDataSet = loadDataSet('./data/dsjtzs_txfz_testB.txt')
normlizedData_test = normalize(testDataSet,1)

# 生成特征
trainData = addSpeedAndSpeedFeat(normlizedData_train)
testData = addSpeedAndSpeedFeat(normlizedData_test)

tempData = trainData

trainData = pd.DataFrame(trainData)
trainData.columns = ['id', 'trail', 'dest', 'label', 
                  'speed', 'speedMax', 'speedMin', 'speedMean', 'speedVar', 'speedUniqueRatio', 'speedMode',
                  'speedDiff', 'speedDiffMax', 'speedDiffMin', 'speedDiffMean', 'speedDiffVar', 'speedDiffLen', 'speedLast',
                  'xSpeed', 'xSpeedMax', 'xSpeedMin', 'xSpeedMean', 'xSpeedVar', 'xSpeedUniqueRatio', 'xSpeedMode',
                  'xSpeedDiff', 'xSpeedDiffMax', 'xSpeedDiffMin', 'xSpeedDiffMean', 'xSpeedDiffVar', 'xSpeedDiffLen', 'xSpeedLast',
                  'ySpeed', 'ySpeedMax', 'ySpeedMean', 'ySpeedVar', 'ySpeedUniqueRatio', 'ySpeedMode',
                  'ySpeedDiff', 'ySpeedDiffMax', 'ySpeedDiffMin', 'ySpeedDiffMean', 'ySpeedDiffVar', 'ySpeedDiffLen', 'ySpeedLast',
                  
                  'oneThirdSpeed', 'oneThirdSpeedMax', 'oneThirdSpeedMin', 'oneThirdSpeedMean', 'oneThirdSpeedVar', 'oneThirdSpeedLen', 'oneThirdSpeedRatio',
                  'xOneThirdSpeed', 'xOneThirdSpeedMax', 'xOneThirdSpeedMin', 'xOneThirdSpeedMean', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen', 'xOneThirdSpeedRatio',
                  'yOneThirdSpeed', 'yOneThirdSpeedMax', 'yOneThirdSpeedMin', 'yOneThirdSpeedMean', 'yOneThirdSpeedVar', 'yOneThirdSpeedLen', 'yOneThirdSpeedRatio',
                  
                  'oneFouthSpeed', 'oneFouthSpeedMax', 'oneFouthSpeedMin', 'oneFouthSpeedMean', 'oneFouthSpeedVar', 'oneFouthSpeedLen', 'oneFouthSpeedRatio',
                  'xOneFouthSpeed', 'xOneFouthSpeedMax', 'xOneFouthSpeedMin', 'xOneFouthSpeedMean', 'xOneFouthSpeedVar', 'xOneFouthSpeedLen', 'xOneFouthSpeedRatio',
                  'yOneFouthSpeed', 'yOneFouthSpeedMax', 'yOneFouthSpeedMin', 'yOneFouthSpeedMean', 'yOneFouthSpeedVar', 'yOneFouthSpeedLen', 'yOneFouthSpeedRatio',
                  
                  'oneFifthSpeed', 'oneFifthSpeedMax', 'oneFifthSpeedMin', 'oneFifthSpeedMean', 'oneFifthSpeedVar', 'oneFifthSpeedLen', 'oneFifthSpeedRatio',
                  'xOneFifthSpeed', 'xOneFifthSpeedMax', 'xOneFifthSpeedMin', 'xOneFifthSpeedMean', 'xOneFifthSpeedVar', 'xOneFifthSpeedLen', 'xOneFifthSpeedRatio',
                  'yOneFifthSpeed', 'yOneFifthSpeedMax', 'yOneFifthSpeedMin', 'yOneFifthSpeedMean', 'yOneFifthSpeedVar', 'yOneFifthSpeedLen', 'yOneFifthSpeedRatio',
                  
                  'accelerate', 'accelerateMax', 'accelerateMean', 'accelerateVar', 'accelerateUniqueRatio', 'accelerateMode',
                  'accelerateDiff', 'accelerateDiffMax', 'accelerateDiffMin', 'accelerateDiffMean', 'accelerateDiffVar', 'accelerateDiffLen', 'accelerateLast',
                  'xAccelerate', 'xAccelerateMax', 'xAccelerateMean', 'xAccelerateVar', 'xAccelerateUniqueRatio', 'xAccelerateMode', 
                  'xAccelerateDiff', 'xAccelerateDiffMax', 'xAccelerateDiffMin', 'xAccelerateDiffMean', 'xAccelerateDiffVar', 'xAccelerateDiffLen', 'xAccelerateLast',
                  'yAccelerate', 'yAccelerateMax', 'yAccelerateMean', 'yAccelerateVar', 'yAccelerateUniqueRatio', 'yAccelerateMode',
                  'yAccelerateDiff', 'yAccelerateDiffMax', 'yAccelerateDiffMin', 'yAccelerateDiffMean', 'yAccelerateDiffVar', 'yAccelerateDiffLen', 'yAccelerateLast',
                  
                  'oneThirdAccelerate', 'oneThirdAccelerateMax', 'oneThirdAccelerateMin', 'oneThirdAccelerateMean', 'oneThirdAccelerateVar', 'oneThirdAccelerateDiffLen',
                  'xOneThirdAccelerate', 'xOneThirdAccelerateMax', 'xOneThirdAccelerateMin', 'xOneThirdAccelerateMean', 'xOneThirdAccelerateVar', 'xOneThirdAccelerateDiffLen',
                  'yOneThirdAccelerate', 'yOneThirdAccelerateMax', 'yOneThirdAccelerateMin', 'yOneThirdAccelerateMean', 'yOneThirdAccelerateVar', 'yOneThirdAccelerateDiffLen',
                  
                  'oneFouthAccelerate', 'oneFouthAccelerateMax', 'oneFouthAccelerateMin', 'oneFouthAccelerateMean', 'oneFouthAccelerateVar', 'oneFouthAccelerateDiffLen',
                  'xOneFouthAccelerate', 'xOneFouthAccelerateMax', 'xOneFouthAccelerateMin', 'xOneFouthAccelerateMean', 'xOneFouthAccelerateVar', 'xOneFouthAccelerateDiffLen',
                  'yOneFouthAccelerate', 'yOneFouthAccelerateMax', 'yOneFouthAccelerateMin', 'yOneFouthAccelerateMean', 'yOneFouthAccelerateVar', 'yOneFouthAccelerateDiffLen',
                  
                  'oneFifthAccelerate', 'oneFifthAccelerateMax', 'oneFifthAccelerateMin', 'oneFifthAccelerateMean', 'oneFifthAccelerateVar', 'oneFifthAccelerateDiffLen',
                  'xOneFifthAccelerate', 'xOneFifthAccelerateMax', 'xOneFifthAccelerateMin', 'xOneFifthAccelerateMean', 'xOneFifthAccelerateVar', 'xOneFifthAccelerateDiffLen',
                  'yOneFifthAccelerate', 'yOneFifthAccelerateMax', 'yOneFifthAccelerateMin', 'yOneFifthAccelerateMean', 'yOneFifthAccelerateVar', 'yOneFifthAccelerateDiffLen',
                  
                  'offSet', 'offSetMax', 'offSetMin', 'offSetMean', 'offSetVar', 'offSetMode', 'offSetModeLastToAim',
                  'xOffSet', 'xOffSetMax', 'xOffSetMin', 'xOffSetMean', 'xOffSetVar', 'xOffSetBackNum', 'xOffSetMode', 'xOffSetLastToAim',
                  'yOffSet', 'yOffSetMax', 'yOffSetMin', 'yOffSetMean', 'yOffSetVar', 'yOffSetBackNum', 'yOffSetMode', 'yOffSetLastToAim',
                  'deltaTime', 'deltaTimeMax', 'deltaTimeMin', 'deltaTimeMean', 'deltaTimeVar', 'deltaTimeUniqueRatio', 'deltaTimeMode', 'deltaTimeTotal',
                  'angle', 'angleMax', 'angleMin', 'angleMean', 'angleVar', 'angleMode', 'angleUniqueRatio',
                  'angleDiff', 'angleDiffMax', 'angleDiffMin', 'angleDiffMean', 'angleDiffVar', 'angleDiffLen', 
                  'angleNoneZero', 'angleNoneZeroMax', 'angleNoneZeroMin', 'angleNoneZeroVar', 'angleNoneZeroMean', 'angleNoneZeroLen',
                  'angleSpeed', 'angleSpeedMax', 'angleSpeedMin', 'angleSpeedMean', 'angleSpeedVar', 'angleSpeedUniqueRatio', 'angleSpeedMode',
                  'angleSpeedDiff', 'angleSpeedDiffMax', 'angleSpeedDiffMin', 'angleSpeedDiffMean', 'angleSpeedDiffVar', 'angleSpeedDiffLen',
                  'angleSpeedNoneZero', 'angleSpeedNoneZeroMax', 'angleSpeedNoneZeroMin', 'angleSpeedNoneZeroVar', 
                  'angleSpeedNoneZeroMean', 'angleSpeedNoneZeroLen',
                  'xCoor', 'xCoorMax', 'xCoorMin', 'xCoorMean', 'xCoorVar', 'xCoorUniqueRatio', 'xCoorMode',
                  'yCoor', 'yCoorMax', 'yCoorMin', 'yCoorMean', 'yCoorVar', 'yCoorUniqueRatio', 'yCoorMode',
                  'time', 'timeMax', 'timeMin', 'timeMean', 'timeVar', 'timeLen',
                  'xAimDist', 'xAimDistMax', 'xAimDistMin', 'xAimDistMean', 'xAimDistVar', 'xAimDistMode', 'xAimDistUniqueRatio',  
                  'yAimDist', 'yAimDistMax', 'yAimDistMin', 'yAimDistMean', 'yAimDistVar', 'yAimDistMode', 'yAimDistUniqueRatio',
                  'aimAngle', 'aimAngleMax', 'aimAngleMin', 'aimAngleMean', 'aimAngleVar', 'aimAngleMode', 'aimAngleUniqueRatio'
                  ]
                  
testData = pd.DataFrame(testData)
testData.columns = ['id', 'trail', 'dest', 
                  'speed', 'speedMax', 'speedMin', 'speedMean', 'speedVar', 'speedUniqueRatio', 'speedMode',
                  'speedDiff', 'speedDiffMax', 'speedDiffMin', 'speedDiffMean', 'speedDiffVar', 'speedDiffLen', 'speedLast',
                  'xSpeed', 'xSpeedMax', 'xSpeedMin', 'xSpeedMean', 'xSpeedVar', 'xSpeedUniqueRatio', 'xSpeedMode',
                  'xSpeedDiff', 'xSpeedDiffMax', 'xSpeedDiffMin', 'xSpeedDiffMean', 'xSpeedDiffVar', 'xSpeedDiffLen', 'xSpeedLast',
                  'ySpeed', 'ySpeedMax', 'ySpeedMean', 'ySpeedVar', 'ySpeedUniqueRatio', 'ySpeedMode',
                  'ySpeedDiff', 'ySpeedDiffMax', 'ySpeedDiffMin', 'ySpeedDiffMean', 'ySpeedDiffVar', 'ySpeedDiffLen', 'ySpeedLast',
                  
                  'oneThirdSpeed', 'oneThirdSpeedMax', 'oneThirdSpeedMin', 'oneThirdSpeedMean', 'oneThirdSpeedVar', 'oneThirdSpeedLen', 'oneThirdSpeedRatio',
                  'xOneThirdSpeed', 'xOneThirdSpeedMax', 'xOneThirdSpeedMin', 'xOneThirdSpeedMean', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen', 'xOneThirdSpeedRatio',
                  'yOneThirdSpeed', 'yOneThirdSpeedMax', 'yOneThirdSpeedMin', 'yOneThirdSpeedMean', 'yOneThirdSpeedVar', 'yOneThirdSpeedLen', 'yOneThirdSpeedRatio',
                  
                  'oneFouthSpeed', 'oneFouthSpeedMax', 'oneFouthSpeedMin', 'oneFouthSpeedMean', 'oneFouthSpeedVar', 'oneFouthSpeedLen', 'oneFouthSpeedRatio',
                  'xOneFouthSpeed', 'xOneFouthSpeedMax', 'xOneFouthSpeedMin', 'xOneFouthSpeedMean', 'xOneFouthSpeedVar', 'xOneFouthSpeedLen', 'xOneFouthSpeedRatio',
                  'yOneFouthSpeed', 'yOneFouthSpeedMax', 'yOneFouthSpeedMin', 'yOneFouthSpeedMean', 'yOneFouthSpeedVar', 'yOneFouthSpeedLen', 'yOneFouthSpeedRatio',
                  
                  'oneFifthSpeed', 'oneFifthSpeedMax', 'oneFifthSpeedMin', 'oneFifthSpeedMean', 'oneFifthSpeedVar', 'oneFifthSpeedLen', 'oneFifthSpeedRatio',
                  'xOneFifthSpeed', 'xOneFifthSpeedMax', 'xOneFifthSpeedMin', 'xOneFifthSpeedMean', 'xOneFifthSpeedVar', 'xOneFifthSpeedLen', 'xOneFifthSpeedRatio',
                  'yOneFifthSpeed', 'yOneFifthSpeedMax', 'yOneFifthSpeedMin', 'yOneFifthSpeedMean', 'yOneFifthSpeedVar', 'yOneFifthSpeedLen', 'yOneFifthSpeedRatio',
                  
                  'accelerate', 'accelerateMax', 'accelerateMean', 'accelerateVar', 'accelerateUniqueRatio', 'accelerateMode',
                  'accelerateDiff', 'accelerateDiffMax', 'accelerateDiffMin', 'accelerateDiffMean', 'accelerateDiffVar', 'accelerateDiffLen', 'accelerateLast',
                  'xAccelerate', 'xAccelerateMax', 'xAccelerateMean', 'xAccelerateVar', 'xAccelerateUniqueRatio', 'xAccelerateMode', 
                  'xAccelerateDiff', 'xAccelerateDiffMax', 'xAccelerateDiffMin', 'xAccelerateDiffMean', 'xAccelerateDiffVar', 'xAccelerateDiffLen', 'xAccelerateLast',
                  'yAccelerate', 'yAccelerateMax', 'yAccelerateMean', 'yAccelerateVar', 'yAccelerateUniqueRatio', 'yAccelerateMode',
                  'yAccelerateDiff', 'yAccelerateDiffMax', 'yAccelerateDiffMin', 'yAccelerateDiffMean', 'yAccelerateDiffVar', 'yAccelerateDiffLen', 'yAccelerateLast',
                  
                  'oneThirdAccelerate', 'oneThirdAccelerateMax', 'oneThirdAccelerateMin', 'oneThirdAccelerateMean', 'oneThirdAccelerateVar', 'oneThirdAccelerateDiffLen',
                  'xOneThirdAccelerate', 'xOneThirdAccelerateMax', 'xOneThirdAccelerateMin', 'xOneThirdAccelerateMean', 'xOneThirdAccelerateVar', 'xOneThirdAccelerateDiffLen',
                  'yOneThirdAccelerate', 'yOneThirdAccelerateMax', 'yOneThirdAccelerateMin', 'yOneThirdAccelerateMean', 'yOneThirdAccelerateVar', 'yOneThirdAccelerateDiffLen',
                  
                  'oneFouthAccelerate', 'oneFouthAccelerateMax', 'oneFouthAccelerateMin', 'oneFouthAccelerateMean', 'oneFouthAccelerateVar', 'oneFouthAccelerateDiffLen',
                  'xOneFouthAccelerate', 'xOneFouthAccelerateMax', 'xOneFouthAccelerateMin', 'xOneFouthAccelerateMean', 'xOneFouthAccelerateVar', 'xOneFouthAccelerateDiffLen',
                  'yOneFouthAccelerate', 'yOneFouthAccelerateMax', 'yOneFouthAccelerateMin', 'yOneFouthAccelerateMean', 'yOneFouthAccelerateVar', 'yOneFouthAccelerateDiffLen',
                  
                  'oneFifthAccelerate', 'oneFifthAccelerateMax', 'oneFifthAccelerateMin', 'oneFifthAccelerateMean', 'oneFifthAccelerateVar', 'oneFifthAccelerateDiffLen',
                  'xOneFifthAccelerate', 'xOneFifthAccelerateMax', 'xOneFifthAccelerateMin', 'xOneFifthAccelerateMean', 'xOneFifthAccelerateVar', 'xOneFifthAccelerateDiffLen',
                  'yOneFifthAccelerate', 'yOneFifthAccelerateMax', 'yOneFifthAccelerateMin', 'yOneFifthAccelerateMean', 'yOneFifthAccelerateVar', 'yOneFifthAccelerateDiffLen',
                  
                  'offSet', 'offSetMax', 'offSetMin', 'offSetMean', 'offSetVar', 'offSetMode', 'offSetModeLastToAim',
                  'xOffSet', 'xOffSetMax', 'xOffSetMin', 'xOffSetMean', 'xOffSetVar', 'xOffSetBackNum', 'xOffSetMode', 'xOffSetLastToAim',
                  'yOffSet', 'yOffSetMax', 'yOffSetMin', 'yOffSetMean', 'yOffSetVar', 'yOffSetBackNum', 'yOffSetMode', 'yOffSetLastToAim',
                  'deltaTime', 'deltaTimeMax', 'deltaTimeMin', 'deltaTimeMean', 'deltaTimeVar', 'deltaTimeUniqueRatio', 'deltaTimeMode', 'deltaTimeTotal',
                  'angle', 'angleMax', 'angleMin', 'angleMean', 'angleVar', 'angleMode', 'angleUniqueRatio',
                  'angleDiff', 'angleDiffMax', 'angleDiffMin', 'angleDiffMean', 'angleDiffVar', 'angleDiffLen', 
                  'angleNoneZero', 'angleNoneZeroMax', 'angleNoneZeroMin', 'angleNoneZeroVar', 'angleNoneZeroMean', 'angleNoneZeroLen',
                  'angleSpeed', 'angleSpeedMax', 'angleSpeedMin', 'angleSpeedMean', 'angleSpeedVar', 'angleSpeedUniqueRatio', 'angleSpeedMode',
                  'angleSpeedDiff', 'angleSpeedDiffMax', 'angleSpeedDiffMin', 'angleSpeedDiffMean', 'angleSpeedDiffVar', 'angleSpeedDiffLen',
                  'angleSpeedNoneZero', 'angleSpeedNoneZeroMax', 'angleSpeedNoneZeroMin', 'angleSpeedNoneZeroVar', 
                  'angleSpeedNoneZeroMean', 'angleSpeedNoneZeroLen',
                  'xCoor', 'xCoorMax', 'xCoorMin', 'xCoorMean', 'xCoorVar', 'xCoorUniqueRatio', 'xCoorMode',
                  'yCoor', 'yCoorMax', 'yCoorMin', 'yCoorMean', 'yCoorVar', 'yCoorUniqueRatio', 'yCoorMode',
                  'time', 'timeMax', 'timeMin', 'timeMean', 'timeVar', 'timeLen',
                  'xAimDist', 'xAimDistMax', 'xAimDistMin', 'xAimDistMean', 'xAimDistVar', 'xAimDistMode', 'xAimDistUniqueRatio',  
                  'yAimDist', 'yAimDistMax', 'yAimDistMin', 'yAimDistMean', 'yAimDistVar', 'yAimDistMode', 'yAimDistUniqueRatio',
                  'aimAngle', 'aimAngleMax', 'aimAngleMin', 'aimAngleMean', 'aimAngleVar', 'aimAngleMode', 'aimAngleUniqueRatio'
                  ]
overSampling = trainData.loc[trainData.label == '0']
for i in range(5):
    trainData = trainData.append(overSampling)

# 分割训练与验证集,训练分类器
'''
predictors = [x for x in trainData.columns if x not in ['id', 'trail', 'dest', 'label',
                                                       'speed', 'xSpeed', 'ySpeed',cc
                                                       'accelerate', 'xAccelerate', 'yAccelerate',
                                                       'offSet', 'xOffSet', 'yOffSet',
                                                       'deltaTime', 'angle', 'xCoor', 'yCoor',
                                                       'speedUniqueRatio', 'xSpeedUniqueRatio', 'ySpeedUniqueRatio',
                                                       'accelerateUniqueRatio', 'xAccelerateUniqueRatio', 'yAccelerateUniqueRatio',
                                                       'deltaTimeUniqueRatio', 'angleUniqueRatio',
                                                       'xCoorUniqueRatio', 'yCoorUniqueRatio',]]
                                                       
predictors = ['speedMax', 'speedMin', 'speedMean', 'speedVar', 'deltaTimeMax', 'deltaTimeMin',
              'deltaTimeVar', 'offSetMax', 'offSetVar', 'angleMax', 'angleMin', 'angleVar', 
              'xCoorMax', 'xCoorMin', 'xCoorVar', 'yCoorMax', 'yCoorMin', 'yCoorVar', 'xOffSetMax', 
              'xOffSetMin', 'xOffSetVar', 'isXIncrease']
                                                     
predictors = ['speedUniqueRatio', 'xSpeedUniqueRatio', 'ySpeedUniqueRatio', 'deltaTimeMax', 'deltaTimeMin',
              'deltaTimeVar', 'deltaTimeUniqueRatio', 'offSetVar', 'xOffSetVar', 'yOffSetVar', 
              'angleMin', 'angleMean', 'angleVar', 'xCoorMax', 'xCoorMin', 'xCoorUniqueRatio', 
              'xOffSetMax', 'xOffSetMin', 'xOffSetVar', 'yOffSetMax', 'yOffSetMin', 'yOffSetVar', 
              'accelerateUniqueRatio', 'xAccelerateUniqueRatio', 'yAccelerateUniqueRatio', 
              'isXIncrease']
'''
              
'''             
predictors = [x for x in trainData.columns if x not in ['trail', 'dest',
                  'speed', 'speedDiff', 'xSpeed', 'xSpeedDiff', 'ySpeed',
                  'ySpeedDiff', 'accelerate', 'accelerateDiff', 'xAccelerate', 
                  'xAccelerateDiff', 'yAccelerate', 'yAccelerateDiff', 'offSet',
                  'xOffSet', 'yOffSet', 'deltaTime', 'angle', 'angleDiff', 'angleSpeed',
                  'angleSpeedDiff', 'xCoor', 'yCoor', 'xAimDist', 'yAimDist',
                  'aimAngle', 'multi', 'div'
                  ]]
''' 
#xgboost特征                 
predictors = ['xCoorMax', 'xCoorMin', 'xCoorVar',
              'timeMax', 'timeMin', 'timeLen', 'timeMean',
              'speedDiffLen', 'speedMax', 'speedMin', 'speedVar',
              'xSpeedMax', 'xSpeedMin', 'xSpeedDiffLen', 'xSpeedDiffVar', 'xSpeedMode',
              'ySpeedMode',
              'oneThirdSpeedMean', 'oneThirdSpeedVar', 'oneThirdSpeedLen', 
              'accelerateMax', 'accelerateDiffMin', 'accelerateVar', 'accelerateMode',
              'accelerateDiffLen', 'accelerateLast',
              'xAccelerateDiffLen', 'xAccelerateVar', 'xAccelerateLast',
              'yAccelerateMode', 'yAccelerateLast',
              'offSetModeLastToAim',
              #'xOffSetBackNum', 
              'angleNoneZeroMax', 'angleNoneZeroMin', 'angleNoneZeroLen',
              'xAimDistMax', 'xAimDistMin',
              'yAimDistMax', 'yAimDistMin',
              'aimAngleMax', 'aimAngleMin',
              ]
              
#xgboost特征                
target = 'label'
predictors = [#第一批数据 179
              'xCoorMin', 'xCoorVar',
              'timeMax', 'timeMin', 'timeLen', 'timeMean',
              'speedDiffLen', 'speedMax', 'speedMin', 'speedVar', 
              'xSpeedMax', 'xSpeedMin', 'xSpeedDiffLen', 'xSpeedDiffVar', 'xSpeedMode',
              'xOneThirdSpeedMax', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen',
              #第二批数据加上 18380 88
              'xOneFouthSpeedVar', 'xOneFouthSpeedLen', 'xOneFouthSpeedRatio',
              'xOneFifthSpeedVar', 'xOneFifthSpeedLen', 'xOneFifthSpeedRatio',
              ]

'''
predictors = ['xCoorMax', 'xCoorMin', #'xCoorVar', 'xCoorUniqueRatio', 'xCoorMode',
              'yCoorMax', 'yCoorMin', #'yCoorVar', 'yCoorUniqueRatio', 'yCoorMode',
              #'speedMax', 'speedMin', 'speedVasr', 'speedUniqueRatio', 'speedMode',
              #'speedMean', 'speedDiffMean', 'speedDiffVar',
              #'speedDiffLen', 'speedLast', 'speedMode',
              'xSpeedMax', 'xSpeedMin', 'xSpeedVar',  'xSpeedUniqueRatio', 'xSpeedMode', 
              'xSpeedMean', 'xSpeedDiffMean','xSpeedDiffLen', 'xSpeedDiffVar', 'xSpeedLast',
              'ySpeedMax', 'ySpeedMean', 'ySpeedVar',  'ySpeedUniqueRatio', 'ySpeedMode',
              'ySpeedDiffMean', 'ySpeedDiffLen', 'ySpeedDiffVar', 'ySpeedLast',
              'oneThirdSpeedMax', 'oneThirdSpeedMin', 'oneThirdSpeedMean', 'oneThirdSpeedVar', 'oneThirdSpeedLen',
              'xOneThirdSpeedMax', 'xOneThirdSpeedMin', 'xOneThirdSpeedMean', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen',
              'yOneThirdSpeedMax', 'yOneThirdSpeedMin', 'yOneThirdSpeedMean', 'yOneThirdSpeedVar', 'yOneThirdSpeedLen',
              #'accelerateMax', 'accelerateMean', 'accelerateVar', 'accelerateUniqueRatio', 'accelerateMode',
              #'accelerateDiffMin', 'accelerateDiffMean', 'accelerateDiffLen', 'accelerateDiffVar', 'accelerateLast',
              'xAccelerateDiffMin', 'xAccelerateDiffLen', 'xAccelerateDiffMean', 'xAccelerateDiffVar', #'xAccelerateLast',
              'xAccelerateMax', 'xAccelerateMean', 'xAccelerateVar', 'xAccelerateUniqueRatio',  'xAccelerateMode',
              'yAccelerateMax', 'yAccelerateMean', 'yAccelerateVar', 'yAccelerateUniqueRatio',  'yAccelerateMode',
              'yAccelerateDiffMean', 'yAccelerateDiffLen', 'yAccelerateDiffMin', 'yAccelerateDiffVar', #'yAccelerateLast',
              #'offSetMax', 'offSetMin', 'offSetVar', 'offSetMode', 'offSetModeLastToAim',
              'xOffSetMax', 'xOffSetMin', 'xOffSetVar', 'xOffSetBackNum', #'xOffSetLastToAim', #'xOffSetMean', 'xOffSetMode',
              'yOffSetMax', 'yOffSetMin', 'yOffSetVar', #'yOffSetLastToAim', #'yOffSetBackNum', #'yOffSetMean', 'yOffSetMode', 
              'angleMax', 'angleMin', 'angleVar', 'angleMode', 'angleUniqueRatio',
              'angleDiffVar', 'angleMean', 'angleDiffMean', 'angleDiffLen', 
              'angleNoneZeroMax', 'angleNoneZeroMin', 'angleNoneZeroVar', 'angleNoneZeroMean', 'angleNoneZeroLen',
              'angleSpeedMax', 'angleSpeedMin', 'angleSpeedVar',  'angleSpeedUniqueRatio', 'angleSpeedMode',
              'angleSpeedDiffMean', 'angleSpeedDiffVar', 'angleSpeedDiffLen', 'angleSpeedMean',
              'angleSpeedNoneZeroMax', 'angleSpeedNoneZeroMin', 'angleSpeedNoneZeroVar', 
              'angleSpeedNoneZeroMean', 'angleSpeedNoneZeroLen',
              'deltaTimeMax', 'deltaTimeMin', 'deltaTimeVar', 'deltaTimeUniqueRatio', 'deltaTimeMode', #'deltaTimeTotal',
              #'timeMax', 'timeMin', 'timeLen', 'timeMean', 'timeVar',
              
              #'xAimDistMax', 'xAimDistMin', 'xAimDistVar', 'xAimDistMode', 'xAimDistUniqueRatio',
              #'yAimDistMax', 'yAimDistMin', 'yAimDistVar', 'yAimDistMode', 'yAimDistUniqueRatio',
              #'aimAngleMax', 'aimAngleMin', 'aimAngleVar', 'aimAngleMode', 'aimAngleUniqueRatio',
              
              'oneThirdSpeed', 'oneThirdSpeedMax', 'oneThirdSpeedMin', 'oneThirdSpeedMean', 'oneThirdSpeedVar', 'oneThirdSpeedLen', 'oneThirdSpeedRatio',
              'xOneThirdSpeed', 'xOneThirdSpeedMax', 'xOneThirdSpeedMin', 'xOneThirdSpeedMean', 'xOneThirdSpeedVar', 'xOneThirdSpeedLen', 'xOneThirdSpeedRatio',
              'yOneThirdSpeed', 'yOneThirdSpeedMax', 'yOneThirdSpeedMin', 'yOneThirdSpeedMean', 'yOneThirdSpeedVar', 'yOneThirdSpeedLen', 'yOneThirdSpeedRatio',
              
              'oneFouthSpeed', 'oneFouthSpeedMax', 'oneFouthSpeedMin', 'oneFouthSpeedMean', 'oneFouthSpeedVar', 'oneFouthSpeedLen', 'oneFouthSpeedRatio',
              'xOneFouthSpeed', 'xOneFouthSpeedMax', 'xOneFouthSpeedMin', 'xOneFouthSpeedMean', 'xOneFouthSpeedVar', 'xOneFouthSpeedLen', 'xOneFouthSpeedRatio',
              'yOneFouthSpeed', 'yOneFouthSpeedMax', 'yOneFouthSpeedMin', 'yOneFouthSpeedMean', 'yOneFouthSpeedVar', 'yOneFouthSpeedLen', 'yOneFouthSpeedRatio',
              
              'oneFifthSpeed', 'oneFifthSpeedMax', 'oneFifthSpeedMin', 'oneFifthSpeedMean', 'oneFifthSpeedVar', 'oneFifthSpeedLen', 'oneFifthSpeedRatio',
              'xOneFifthSpeed', 'xOneFifthSpeedMax', 'xOneFifthSpeedMin', 'xOneFifthSpeedMean', 'xOneFifthSpeedVar', 'xOneFifthSpeedLen', 'xOneFifthSpeedRatio',
              'yOneFifthSpeed', 'yOneFifthSpeedMax', 'yOneFifthSpeedMin', 'yOneFifthSpeedMean', 'yOneFifthSpeedVar', 'yOneFifthSpeedLen', 'yOneFifthSpeedRatio',
              
              'oneThirdAccelerate', 'oneThirdAccelerateMax', 'oneThirdAccelerateMin', 'oneThirdAccelerateMean', 'oneThirdAccelerateVar', 'oneThirdAccelerateDiffLen',
              'xOneThirdAccelerate', 'xOneThirdAccelerateMax', 'xOneThirdAccelerateMin', 'xOneThirdAccelerateMean', 'xOneThirdAccelerateVar', 'xOneThirdAccelerateDiffLen',
              'yOneThirdAccelerate', 'yOneThirdAccelerateMax', 'yOneThirdAccelerateMin', 'yOneThirdAccelerateMean', 'yOneThirdAccelerateVar', 'yOneThirdAccelerateDiffLen',

              'oneFouthAccelerate', 'oneFouthAccelerateMax', 'oneFouthAccelerateMin', 'oneFouthAccelerateMean', 'oneFouthAccelerateVar', 'oneFouthAccelerateDiffLen',
              'xOneFouthAccelerate', 'xOneFouthAccelerateMax', 'xOneFouthAccelerateMin', 'xOneFouthAccelerateMean', 'xOneFouthAccelerateVar', 'xOneFouthAccelerateDiffLen',
              'yOneFouthAccelerate', 'yOneFouthAccelerateMax', 'yOneFouthAccelerateMin', 'yOneFouthAccelerateMean', 'yOneFouthAccelerateVar', 'yOneFouthAccelerateDiffLen',
              
              'oneFifthAccelerate', 'oneFifthAccelerateMax', 'oneFifthAccelerateMin', 'oneFifthAccelerateMean', 'oneFifthAccelerateVar', 'oneFifthAccelerateDiffLen',
              'xOneFifthAccelerate', 'xOneFifthAccelerateMax', 'xOneFifthAccelerateMin', 'xOneFifthAccelerateMean', 'xOneFifthAccelerateVar', 'xOneFifthAccelerateDiffLen',
              'yOneFifthAccelerate', 'yOneFifthAccelerateMax', 'yOneFifthAccelerateMin', 'yOneFifthAccelerateMean', 'yOneFifthAccelerateVar', 'yOneFifthAccelerateDiffLen',              
              ]  
'''
        
'''
train = trainData[predictors]
test = testData[predictors]
#将特征进行融合
preLen = len(predictors)
for i in range(0, preLen - 1):
    for j in range(i+1, preLen):
        newColumn = predictors[i] + '+' + predictors[j]
        train[newColumn] = train[predictors[i]] + train[predictors[j]]
        test[newColumn] = test[predictors[i]] + test[predictors[j]]
        newColumn = predictors[i] + '-' + predictors[j]
        train[newColumn] = train[predictors[i]] - train[predictors[j]]
        test[newColumn] = test[predictors[i]] - test[predictors[j]]
        newColumn = predictors[i] + '*' + predictors[j]
        train[newColumn] = train[predictors[i]] * train[predictors[j]]
        test[newColumn] = test[predictors[i]] * test[predictors[j]]
        newColumn = predictors[i] + '/' + predictors[j]
        train[newColumn] = train[predictors[i]] / train[predictors[j]]
        test[newColumn] = test[predictors[i]] / test[predictors[j]]

train = train.fillna(-1)
train = train.replace(np.inf, -3)
train.to_csv('./train.csv', index=False)

test = test.fillna(-1)
test = test.replace(np.inf, -3)
test.to_csv('./test.csv', index=False)
'''

#
#clf = svm.SVC()
#scores = cv.cross_val_score(clf, trainData[predictors], trainData[target], cv = 10)
#clf.fit(trainData[predictors], trainData[target])
#y_pred = clf.predict(testData[predictors])
#
#
##clf = LogisticRegression()
##clf = GradientBoostingClassifier(n_estimators=300,random_state=2017)
#clf = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20,max_depth=3,max_features='sqrt' ,random_state=75)
#scores = cv.cross_val_score(clf, trainData[predictors], trainData[target], cv = 3)
#clf.fit(trainData[predictors], trainData[target])
#y = clf.predict(testData[predictors])
#y_score = clf.predict_proba(testData[predictors])
#
#clf = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20,max_depth=3,max_features='sqrt' ,random_state=75)
#scores = cv.cross_val_score(clf, trainData[predictors], trainData[target], cv = 3)
#clf.fit(trainData[predictors], trainData[target])
#y = clf.predict(testData1[predictors])
#y_score = clf.predict_proba(testData[predictors])

'''
subTrainData.columns = ['subtrainData_1', 'subtrainData_2', 'subtrainData_3',
                        'subtrainData_4', 'subtrainData_5', 'subtrainData_6',
                        'subtrainData_7', 'subtrainData_8', 'subtrainData_9',
                        'subtrainData_10', 'subtrainData_11', 'subtrainData_12',
                        'subtrainData_13', 'subtrainData_14', 'subtrainData_15',
                        'subtrainData_16', 'subtrainData_17', 'subtrainData_18',
                        'subtrainData_19', 'subtrainData_20']


instanceIDs = testData.id
res = instanceIDs.to_frame()
res['prob_0'] = y_score[:, 0]
res['prob_1'] = y_score[:, 1]
res['id'] = res['id'].astype(int)
res = res.sort_values(by='prob_0')
res[80000:].id.to_csv('./BDC20160720.txt', header=None, index=False)

# lgb
# create dataset for lightgbm
lgb_train = lgb.Dataset(trainData[predictors], trainData[target])

params = {
            'boosting_type': 'gbdt',
            #'objective': 'binary',
            #'metric': ['binary_logloss', 'auc'],
            #'metric': 'deviance',
            #'num_leaves': 7,
            #'learning_rate': 0.05,
            #'learning_rate' : 0.1,
            #'feature_fraction': 0.83,
            #'bagging_fraction': 0.85,
            #'bagging_freq': 5,
            #'verbose': 0
    }

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=300
                )

y = gbm.predict(testData[predictors])

instanceIDs = testData.id
res = instanceIDs.to_frame()
res['prob'] = y
res['id'] = res['id'].astype(int)
res = res.sort_values(by='prob')  
res.iloc[0:20000].id.to_csv('./BDC20160715.txt', header=None, index=False)

#average = 0
#for i in range(5):
#    x_train, x_test, y_train, y_test\+6
#        = train_test_split(trainData[predictors], trainData[target], test_size = 0.2)
#    clf = LogisticRegression()  
#    clf.fit(x_train, y_train)  
#    y_pred = clf.predict(x_test)
#    p = np.mean(y_pred == y_test)
#    print p
#    average += p

speed = []
f = open('./speed.txt')
for line in f.readlines():
    speed.append(line)

speedResult = []
for i in range(len(speed)):
    print i
    temp = speed[i].strip()
    temp = temp.strip('[')
    temp = temp.strip(']')
    temp = temp.split(',')
    if temp[0] == '':
        continue
        speedResult.append([])
    temp = map(float, temp)
    speedResult.append(temp)
'''
trainData = pd.read_csv('trainData1.csv')
testData = pd.read_csv('testData1.csv')
#xgb
#dtrain = xgb.DMatrix(trainData[predictors], label=trainData[target])
#dtest = xgb.DMatrix(testData[predictors])
#param = {'booster':'gbtree', 'objective':'binary:logistic', 'eta':0.25,
#         'gamma':0.00000001, 'min_child_weight':10, 'max_depth':6,
#         'subsample':0.5, 'colsample_bytree':0.8, 'eval_metric':"logloss",
#         'eval_metric':"auc", 'scale_pos_weight':1, 'seed':0}
##param = {}
#numRound = 88
#clf = xgb.train(param, dtrain, numRound) 
#y = clf.predict(dtest)
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

dtrain = xgb.DMatrix(trainData[predictors], label=trainData[target])
dtest = xgb.DMatrix(testData1[predictors])
param = {'booster':'gbtree', 
         'objective':'binary:logistic', 
         'eta':0.25,
         'gamma':0.00000001, 
         'min_child_weight':10, 
         'max_depth':6,
         'subsample':0.5, 
         'colsample_bytree':0.8, 
         'eval_metric':"logloss",
         'eval_metric':"auc", 
         'scale_pos_weight':1, 
         'seed':0
         }
#param = {}
#numRound = 88
clf = xgb.train(param, dtrain, numRound) 
y = clf.predict(dtest)

instanceIDs = testData.id
res = instanceIDs.to_frame()
res['prob'] = y
res['id'] = res['id'].astype(int)
res = res.sort_values(by='prob')  
res.iloc[0:20000].id.to_csv('./BDC20160715.txt', header=None, index=False)

