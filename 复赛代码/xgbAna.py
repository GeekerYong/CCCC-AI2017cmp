# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:31:37 2017

@author: sw
"""

import sys
from pyspark import SparkContext
import numpy as np
from collections import Counter

output = sys.argv[1]
fold1input = sys.argv[2]
fold2input = sys.argv[3]
fold3input = sys.argv[4]
fold4input = sys.argv[5]
fold5input = sys.argv[6]

sc = SparkContext(appName="MTRCcmpxgbAna")
f1 = sc.textFile(fold1input)
f2 = sc.textFile(fold2input)
f3 = sc.textFile(fold3input)
f4 = sc.textFile(fold4input)
f5 = sc.textFile(fold5input)

def ana(data):
    result_final = []
    for i, item in enumerate(data):
        item = float(item.split('\t')[1])
        if item<0.5:
            result_final.append(0)
        else:
            result_final.append(1)

result1 = ana(f5.collect())
result2 = ana(f4.collect())
result3 = ana(f3.collect())
result4 = ana(f2.collect())
result5 = ana(f1.collect())

final_result = result1 +result2 +result3 +result4 +result5
print final_result
