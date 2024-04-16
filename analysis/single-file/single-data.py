# -*- coding: utf8 -*-

import os  # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# absolute current file path
# os.chdir(r'C:\Users\sergi\repositorios\gunn-diode-deeplearning-tfg\analysis\single-file')

# relative current file path
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# read the data file
datafile = pd.read_fwf("corr01.txt",colspecs=[(0,12),(13,25),(26,38),(39,51),(52,64),(65,77)],names=['Time','V1','V2','V3','V4','V5']);

# print(datafile.iloc[0:10])   # the same as the later one
print("\n", datafile.head(10))