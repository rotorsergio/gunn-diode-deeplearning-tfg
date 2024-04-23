# -*- coding: utf8 -*-

import time
import os # for file management
from turtle import st # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
from scipy.fft import fft,fftfreq # scientific computing (used for fourier transform)

start_time = time.time()
file_identifier = '02' # file identifier

print("\nSINGLE DATA FILE ANALYSIS\n")
print("CURRENT DENSITY OSCILLATIONS IN GUNN DIODE\n")
print("------------------------------------------\n")
print("Reading and processing data file...\n")

# relative current file path setting
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

#NEEDED FILE NAMES

datafile_name = file_identifier + 'corr.txt' # data file name
plot_name = file_identifier + 'plot.png' # plot name
fourier_name = file_identifier + 'fourier.png' # fourier plot name

# read the data file
# datafile = pd.read_fwf("corr01.txt",colspecs=[(0,12),(13,25),(26,38),(39,51),(52,64),(65,77)],names=['V0','V1','V2','V3','V4','V5'])

# this read only gets the 4th column as it seems all other a irrelevant or redundant
dataframe = pd.read_fwf(datafile_name,colspecs=[(39,51)],names=['Current']) # read the dataframe

valid_data = dataframe.loc[dataframe.index >= 200000].copy() # ignore thermalization data, copy to avoid SettingWithCopyWarning

# transform row index to time in fs
valid_data['Time'] = valid_data.index * 0.2 # 0.2 fs per row
valid_data = valid_data[['Time', 'Current']]  # keep only the Time and Current columns, and exchange order

print("File " + datafile_name + " processed successfully. Showing some statistics and some rows:\n")

print("\n", valid_data.describe())  # print some statistics
print("\n", valid_data)  # print the datafile

print("\nPlotting the data...")

# plot the datafile
first_plot = valid_data.plot(
    x='Time',
    y='Current',
    title='Current density in Gunn diode',
    xlabel='Time (fs)',
    ylabel='Current density',
    xlim=([40e3,100e3]),  # limit x axis
    figsize=(16,9),
    style='r-',  # red color, solid line
    grid=True,
    legend=False
)
plt.savefig(plot_name) # save the plot as a png file

print("Plot saved as " + plot_name + "\n")

# FOURIER TRANSFORM

print("\nPerforming Fourier transform...\n")
N=len(valid_data) # number of data points
dt=0.2 # 0.2 fs time step
print("Number of data points: ", N)
print("Time step: ", dt)

currents_array=valid_data['Current'].to_numpy() # convert the current density to a numpy array, as a dataframe column is not a valid input for the fft function
# dc_current = 15000
dc_current = np.mean(currents_array) # calculate the mean value of the current density
oscillation_array = currents_array - dc_current # remove the mean value from the current density

print("Mean current density: ", dc_current)

fft_amplitude =fft(oscillation_array)[:N//2] # fourier transform of the current density
fft_freq = fftfreq(N, dt)[:N//2] # fourier frequencies
# [:N/2] is used to keep only the positive frequencies, as the fft function returns the negative frequencies after the positive ones, and are redundant (we are dealing with real data)
current_PSD = ((np.abs(fft_amplitude))**2)*2//N # type: ignore # power spectral density, normalized (single sided,  N datapoints)

fourier = pd.DataFrame({'Frequency':fft_freq*1e3, 'Amplitude':current_PSD}) # type: ignore # create a dataframe with the fourier data

print("Fourier transform performed successfully. Showing some statistics and some rows:\n")

print("\n", fourier.describe())  # print some statistics
print("\n", fourier)  # print the fourier data

print("\nPlotting the Fourier transform...")

plot_focus=16 # focus on the first 10 THz

fourier_plot, axs=plt.subplots(2,figsize=(16,9)) # create a figure with 2 subplots

fourier.plot(
    ax=axs[0], # plot the fourier data in the first subplot
    x='Frequency',
    y='Amplitude',
    title='Fourier transform of current density in Gunn diode',
    xlabel='Frequency (THz)',
    ylabel='Current PSD',
    xlim=([0, plot_focus]),  # limit x axis
    style='b-',  # blue color, dotted line with circles
    grid=True,
    legend=False,
)
fourier.plot(
    ax=axs[1], # plot the fourier data in the second subplot
    x='Frequency',
    y='Amplitude',  
    title='Fourier transform of current density in Gunn diode',
    xlabel='Frequency (THz)',
    ylabel='Current PSD (log scale)',
    xlim=([0, plot_focus]),  # limit x axis
    logy=True,  # log scale for y axis
    style='b-',  # blue color, dotted line with circles
    grid=True,
    legend=False,
)
plt.tight_layout()
plt.savefig(fourier_name) # save the fourier plot as a png file

print("Fourier plot saved as " + fourier_name + "\n")

timeit = time.time() - start_time
print("\nAnalysis completed in ", timeit, " seconds.\n")