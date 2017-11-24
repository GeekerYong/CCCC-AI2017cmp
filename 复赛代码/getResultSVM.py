# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 22:51:04 2017

@author: sw
"""
import sys
from pyspark import SparkContext
import math
import numpy as np
from collections import Counter

output = sys.argv[1]
input = sys.argv[2]

sc = SparkContext()
result = sc.textFile(input).collect()
print result
result_final = []

for i, item in enumerate(result):
    print item
    if item == 0:
        result_final.append(i+1)
print len(result_final)
#result_rdd = sc.parallelize(result_final)
#print len(result_final)
#result_rdd.saveAsTextFile(output)