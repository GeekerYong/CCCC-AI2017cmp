# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 16:01:00 2017

@author: sw
"""

import sys
from operator import add
import numpy as np
from pyspark import SparkContext
from myutil import normalize,generate_features

output1 = sys.argv[1]
output2 = sys.argv[1]
output3 = sys.argv[1]
output4 = sys.argv[1]
output5 = sys.argv[1]
output11 = sys.argv[1]
output22 = sys.argv[1]
output33 = sys.argv[1]
output44 = sys.argv[1]
output55 = sys.argv[1]
input = sys.argv[2]

#
#data = ["1 1:1.26666666667 2:0.0134874759152 3:0.553955627607 4:16 5:3.26666666667 6:0.0134874759152 7:0.553955627607 8:0.816666666667 9:16 10:3.26666666667 11:0.614527689655",
#        "0 1:2.26666666667 2:0.0134874759152 3:0.553955627607 4:16 5:3.26666666667 6:0.0134874759152 7:0.553955627607 8:0.816666666667 9:16 10:3.26666666667 11:0.614527689655",
#        "1 1:3.26666666667 2:0.0134874759152 3:0.553955627607 4:16 5:3.26666666667 6:0.0134874759152 7:0.553955627607 8:0.816666666667 9:16 10:3.26666666667 11:0.614527689655",
#        "0 1:4.26666666667 2:0.0134874759152 3:0.553955627607 4:16 5:3.26666666667 6:0.0134874759152 7:0.553955627607 8:0.816666666667 9:16 10:3.26666666667 11:0.614527689655",
#        "1 1:5.26666666667 2:0.0134874759152 3:0.553955627607 4:16 5:3.26666666667 6:0.0134874759152 7:0.553955627607 8:0.816666666667 9:16 10:3.26666666667 11:0.614527689655"
#        ]
sc = SparkContext(appName="MTRCcmpstackingSplit")
trainDataSet = sc.textFile(input) ##线上
trainDataList = trainDataSet.collect()
#trainDataList = data
#trainDataList = [x.split(' ') for x in trainDataList]
#trainNor = []
#for sample in trainDataList:
#    tempData = []
#    for item in sample:
#        if item =='0' or item == '1':
#            tempData.append(int(item))
#        else:
#            temp  =item.split(':')[1]
#            tempData.append(float(temp))
#    trainNor.append(tempData)
#trainArr = np.array(trainNor)
trainArr = np.array(trainDataList)
#data1,data2,data3,data4,data5 = np.split(trainArr, 5)
##
#def nor(data):
#    nor_data = []
#    for block in data:
#        for item in block:
#            print item
#            nor_data.append(' '.join(item))
#    return nor_data
#
#trainFold1 = np.vstack((data1,data2,data3,data4))
#trainFold1 = nor(trainFold1)
#testFold1 =  data5
#testFold1 = nor(testFold1)
#
#trainFold2 =np.vstack((data1,data2,data3,data5))
#trainFold2 = nor(trainFold2)
#testFold2 =  data4
#testFold2 = nor(testFold2)
#
#trainFold3 = np.vstack((data1,data2,data4,data5))
#trainFold3 = nor(trainFold3)
#testFold3 =  data3
#testFold3 = nor(testFold3)
#
#trainFold4 =np.vstack((data1,data3,data4,data5))
#trainFold4 = nor(trainFold4)
#testFold4 =  data2
#testFold4 = nor(testFold4)
#
#trainFold5 =np.vstack((data2,data3,data4,data5))
#trainFold5 = nor(trainFold5)
#testFold5 =  data1
#testFold5 = nor(testFold5)
#
#print len(trainFold1)
#
#print len(testFold1)
data1,data2,data3,data4,data5 = np.split(trainArr, 5)
data1 = data1.tolist()
data2 = data2.tolist()
data3 = data3.tolist()
data4 = data4.tolist()
data5 = data5.tolist()

trainFold1 = data1+data2+data3+data4
testFold1 =  data5
print "trainFold1"
print len(trainFold1)
print trainFold1
print "testFold1"
print len(testFold1)
print testFold1

trainFold2 =data1+data2+data3+data5
testFold2 =  data4


trainFold2 =data1+data2+data4+data5
testFold2 =  data3

trainFold2 =data1+data3+data4+data5
testFold2 =  data2

trainFold2 =data2+data3+data4+data5
testFold2 =  data1

trainFold1 = sc.parallelize(trainFold1)
testFold1 = sc.parallelize(testFold1)
trainFold1.saveAsTextFile(output1)
testFold1.saveAsTextFile(output11)

trainFold2 = sc.parallelize(trainFold2)
testFold2 = sc.parallelize(testFold2)
trainFold2.saveAsTextFile(output2)
testFold2.saveAsTextFile(output22)

trainFold3 = sc.parallelize(trainFold3)
testFold3 = sc.parallelize(testFold3)
trainFold3.saveAsTextFile(output3)
testFold3.saveAsTextFile(output33)

trainFold4 = sc.parallelize(trainFold4)
testFold4 = sc.parallelize(testFold4)
trainFold4.saveAsTextFile(output4)
testFold4.saveAsTextFile(output44)

trainFold5 = sc.parallelize(trainFold5)
testFold5 = sc.parallelize(testFold5)
trainFold5.saveAsTextFile(output5)
testFold5.saveAsTextFile(output55)