# -*- coding: utf8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

datafile = open("corr01.txt","r")

for i in range(10):
    print(datafile.readline())

datafile.close()