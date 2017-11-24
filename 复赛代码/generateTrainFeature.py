# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 18:04:03 2017

@author: sw
"""

import sys
from pyspark import SparkContext
import math
import numpy as np
from collections import Counter

output = sys.argv[1]
Xgbinput = sys.argv[2]
Rfinput = sys.argv[3]

sc = SparkContext(appName="MTRCcmpgenerTF2nd")

xgbTrain = sc.textFile(Xgbinput)
xgbTrain = xgbTrain.collect()
rfTrain = sc.textFile(Rfinput)
rfTrain = rfTrain.collect()

print xgbTrain
print rfTrain