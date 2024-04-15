# -*- coding: utf8 -*-

import numpy as np

datafile = open("corr01.txt","r")

for i in range(10):
    print(datafile.readline())

datafile.close()