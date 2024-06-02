# -*- coding: utf8 -*-

import time
import os # for file management
from turtle import st # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
from scipy.fft import fft,fftfreq # scientific computing (used for fourier transform)

start_time = time.time()

# PARAMETERS ======================================================================================

file_id = '01' # file identifier
dt=1 # 1 fs time step
window_size = 2000 # window size for smoothing
plot_focus=2e-1 # focus on the first 10 GHz, set to .5 for max range
figure_size=(16,9) # figure size

# =================================================================================================

# DEFINED FUNCTIONS ===============================================================================

def set_path(): # relative current file path setting
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    return "Path set successfully."

def setup_names(file_id):
    global datafile_name, plot_name, fourier_name
    datafile_name = file_id + 'corr.txt'
    plot_name = file_id + 'plot.png'
    fourier_name = file_id + 'fourier.png'
    return "File names set successfully."

def power_spectrum_density(data_array, N, dt):
    fft_amplitude = fft(data_array)[:N//2] # fourier transform of the current density
    fft_freq = fftfreq(N, dt)[:N//2] # fourier frequencies
    array_PSD = ((np.abs(fft_amplitude))**2)*2//N # type: ignore # power spectral density, normalized (single sided,  N datapoints)
    return array_PSD, fft_freq

plot_config = {
    'figsize': figure_size,
    'style': ['b:','r-'],
    'grid': True
}
# =================================================================================================

# MAIN CODE

print("\nSINGLE DATA FILE ANALYSIS\n")
print("CURRENT DENSITY OSCILLATIONS IN GUNN DIODE\n")
print("------------------------------------------\n")
print("Reading and processing data file...\n")

set_path() # set the path to the current file location
setup_names(file_id) # set the file names

dataframe = pd.read_fwf(datafile_name,colspecs=[(39,51)],names=['Current']) # read the dataframe
# only gets the 4th column, which is the drain electron current density

valid_data = dataframe.loc[dataframe.index >= 200000].copy() # ignore thermalization data, copy to avoid SettingWithCopyWarning

# transform row index to time in fs
valid_data['Time'] = valid_data.index*dt # time in fs
valid_data = valid_data[['Time', 'Current']]  # keep only the Time and Current columns, and exchange order

# SMOOTHING THE DATA (rolling mean)
print("\nSmoothing the data...\n")
valid_data['Smoothened_Current'] = valid_data['Current'].rolling(window=window_size).mean()
valid_data = valid_data.dropna()
# valid_data['Smoothed_Current'] = valid_data['Smoothed_Current'].fillna(valid_data['Current'].mean()) # fill the first half of the data with the original data

print("File " + datafile_name + " processed successfully. Showing some statistics and some rows:\n")
print("\n", valid_data.describe())  # print some statistics
print("\n", valid_data)  # print the datafile

# MODULATION INDEX
print("\nCalculating the modulation index...")
max_current = valid_data['Smoothened_Current'].max() # maximum current density
min_current = valid_data['Smoothened_Current'].min() # minimum current density
modulation_index = (max_current - min_current) / ( valid_data['Smoothened_Current'].mean()) # modulation index
print("Modulation index: ", modulation_index, "mA/nm\n")

# plot the datafile
print("\nPlotting the data...")
first_plot, axs = plt.subplots(2, figsize=figure_size) # create a figure with 1 subplot

N_current=len(valid_data) # number of data points
N_smoothened=len(valid_data['Smoothened_Current']) # number of data points in the smoothed data
print("Number of data points: ", N_current)
print("Number of smoothened data points: ", N_smoothened)
print("Time step: ", dt, " fs")

currents_array=valid_data['Current'].to_numpy() # convert the current density to a numpy array, as a dataframe column is not a valid input for the fft function
smoothened_array=valid_data['Smoothened_Current'].to_numpy() # convert the smoothed current density to a numpy array
dc_current = np.mean(currents_array) # calculate the mean value of the current density
dc_smoothened = np.mean(smoothened_array) # calculate the mean value of the smoothed current density
oscillation_array = currents_array - dc_current # remove the mean value from the current density
smoothened_oscillation_array = smoothened_array - dc_smoothened # remove the mean value from the smoothed current density

print("Mean current density: ", dc_current)

oscillation_df=pd.DataFrame({'Time':valid_data['Time'],'Oscillation':oscillation_array, 'Smoothened_Oscillation':smoothened_oscillation_array}) # type: ignore # create a dataframe with the oscillation data

first_plot = valid_data.plot(
    ax=axs[0], # plot the data in the first subplot
    x='Time',
    y=['Current','Smoothened_Current'],
    title='Current density in Gunn diode',
    xlabel='Time (fs)',
    ylabel=' Drain current density',
    xlim=([2e5,5e5]),  # limit x axis
    **plot_config
)
first_plot = oscillation_df.plot(
    ax=axs[1], # plot the data in the second subplot
    x='Time',
    y=['Oscillation','Smoothened_Oscillation'],
    title='Current density in Gunn diode, without offset',
    xlabel='Time (fs)',
    ylabel='Drain current density',
    xlim=([2e5,5e5]),  # limit x axis
    **plot_config
)
plt.tight_layout()
plt.savefig(plot_name) # save the plot as a png file

print("Plot saved as " + plot_name + "\n")

# FOURIER TRANSFORM

print("\nPerforming Fourier transform...\n")

current_PSD, current_freq = power_spectrum_density(oscillation_array, N_current, dt) # power spectral density of the current density
smoothed_PSD, smoothed_freq = power_spectrum_density(smoothened_oscillation_array, N_smoothened, dt) # power spectral density of the smoothed current density

PSD_df = pd.DataFrame({'Frequency':current_freq, 'Amplitude':current_PSD, 'Smoothed amplitude':smoothed_PSD}) # type: ignore # create a dataframe with the fourier data

print("Fourier transform performed successfully. Showing some statistics and some rows:\n")

print("\n", PSD_df.describe())  # print some statistics
print("\n", PSD_df)  # print the fourier data

print("\nPlotting the Fourier transform...")


fourier_plot, axs=plt.subplots(2,figsize=(16,9)) # create a figure with 2 subplots

PSD_df.plot(
    ax=axs[0], # plot the fourier data in the first subplot
    x='Frequency',
    y=['Amplitude','Smoothed amplitude'],
    title='Fourier transform of current density in Gunn diode',
    xlabel='Frequency (PHz)',
    ylabel='Current PSD',
    **plot_config
)
PSD_df.plot(
    ax=axs[1], # plot the fourier data in the second subplot
    x='Frequency',
    y=['Amplitude','Smoothed amplitude'],
    title='Fourier transform of current density in Gunn diode',
    xlabel='Frequency (THz)',
    ylabel='Current PSD (log scale)',
    xlim=([0, plot_focus]),  # limit x axis
    logy=True,  # log scale for y axis
    **plot_config
)
plt.tight_layout()
plt.savefig(fourier_name) # save the fourier plot as a png file

print("Fourier plot saved as " + fourier_name + "\n")

timeit = time.time() - start_time
print("\nAnalysis completed in ", timeit, " seconds.\n")