# -*- coding: utf8 -*-

import os # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.fft import fft,fftfreq # scientific computing (used for fourier transform)
import matplotlib.pyplot as plt # for plotting

print("\nSINGLE DATA FILE ANALYSIS\n")
print("CURRENT DENSITY OSCILLATIONS IN GUNN DIODE\n")
print("------------------------------------------\n")
print("Reading and processing data file...\n")

# relative current file path setting
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# read the data file
# datafile = pd.read_fwf("corr01.txt",colspecs=[(0,12),(13,25),(26,38),(39,51),(52,64),(65,77)],names=['V0','V1','V2','V3','V4','V5'])

# this read only gets the 4th column as it seems all other a irrelevant or redundant
datafile = pd.read_fwf('corr01.txt',colspecs=[(39,51)],names=['Current']) # read the data file

valid_data = datafile.loc[datafile.index >= 200000].copy() # ignore thermalization data

# transform row index to time in fs
valid_data['Time'] = valid_data.index * 0.2 # 0.2 fs per row
valid_data = valid_data[['Time', 'Current']]  # keep only the Time and Current columns, and exchange order

print("File processed successfully. Showing some statistics and some rows:\n")

print("\n", valid_data.describe())  # print some statistics
print("\n", valid_data.head(5))  # print the first 5 rows
print("\n", valid_data.tail(5))  # print the last 5 rows

print("\nPlotting the data...")

# plot the datafile
first_plot = valid_data.plot(
    x='Time',
    y='Current',
    title='Current density in Gunn diode',
    xlabel='Time (fs)',
    ylabel='Current density',
    figsize=(16,9),
    style='r-',  # red color, solid line
    grid=True,
    legend=False,
    xlim=([40e3,100e3])  # limit x axis
)
plt.savefig('plot.png')

print("Plot saved as plot.png\n")

# FOURIER TRANSFORM

print("\nPerforming Fourier transform...\n")
N=len(valid_data) # number of data points
dt=0.2 # 0.2 fs time step
print("Number of data points: ", N)
print("Time step: ", dt)

currents_array=valid_data['Current'].to_numpy() # convert the current density to a numpy array, as a dataframe column is not a valid input for the fft function

fft_amplitude =fft(currents_array)[:N//2] # fourier transform of the current density
fft_freq = fftfreq(N, dt)[:N//2] # fourier frequencies
# [:N//2] is used to keep only the positive frequencies, as the fft function returns the negative frequencies after the positive ones

fourier = pd.DataFrame({'Frequency':fft_freq, 'Amplitude':np.abs(fft_amplitude)}) # type: ignore # create a dataframe with the fourier data

print("Fourier transform performed successfully. Showing some statistics and some rows:\n")

print("\n", fourier.describe())  # print some statistics
print("\n", fourier.head(5))  # print the first 5 rows
print("\n", fourier.tail(5))  # print the last 5 rows

print("\nPlotting the Fourier transform...")

fourier_plot=fourier.plot(
    x='Frequency',
    y='Amplitude',
    title='Fourier transform of current density in Gunn diode',
    xlabel='Frequency (PHz)',
    ylabel='Amplitude',
    figsize=(16,9),
    style='b-',  # blue color, solid line
    grid=True,
    legend=False,
    xlim=([0,1e-3])  # limit x axis
)
plt.savefig('fourier.png')

print("Fourier plot saved as fourier.png\n")