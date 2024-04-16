# -*- coding: utf8 -*-

import os  # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp # scientific computing (used for fourier transform)
import matplotlib.pyplot as plt # for plotting

# absolute current file path
# os.chdir(r'C:\Users\sergi\repositorios\gunn-diode-deeplearning-tfg\analysis\single-file')

# relative current file path setting
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# read the data file
# datafile = pd.read_fwf("corr01.txt",colspecs=[(0,12),(13,25),(26,38),(39,51),(52,64),(65,77)],names=['Time','V1','V2','V3','V4','V5']);
# this read only gets the 4th column as it seems all other a irrelevant or redundant
datafile = pd.read_fwf("corr01.txt",colspecs=[(39,51)],names=['correlation']);

# print("\n", datafile.head(10))  # print the first 10 rows
# print("\n", datafile.iloc[200000:200100])   # print rows from a to b with iloc[a:b]

valid_data = datafile.loc[datafile.index >= 200000] # ignore thermalization data

first_plot = valid_data.plot(
    title="Correlation data",
    xlabel="Time (fs)",
    ylabel="Correlation",
    figsize=(16,9),
    style='r-',  # red color, circle markers, solid line
    grid=True,
)

plt.savefig("correlation.png")