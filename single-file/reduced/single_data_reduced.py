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

file_id = '03' # file identifier
dt=1 # 1 fs time step
window_size = 2000 # window size for smoothing
plot_focus=2e-1 # focus on the first 10 GHz, set to .5 for max range
figure_size=(16,9) # figure size
perform_fourier = False

# =================================================================================================

# DEFINED FUNCTIONS ===============================================================================

def set_path(): # relative current file path setting
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    return "Path set successfully."

def setup_names(file_id):
    global datafile_name, plot_name, fourier_name
    datafile_name = 'corr_reduced'
    plot_name = f"{file_id}plot.png"
    fourier_name = f"{file_id}fourier.png"
    return "File names set successfully."

def smoothening(valid_data,window_size):
    # Adds a smoothened column by rolling mean method
    print("Smoothening the data...\n")
    valid_data['Smoothened_Current'] = valid_data['Current'].rolling(window=window_size).mean()
    valid_data = valid_data.dropna()
    return valid_data

def modulation_index(data):
    print("\nCalculating the modulation index...")
    max_current = data['Smoothened_Current'].max() # maximum current density
    min_current = data['Smoothened_Current'].min() # minimum current density
    modulation_index = (max_current - min_current) / ( data['Smoothened_Current'].mean()) # modulation index
    print(f"Modulation index: {round(modulation_index)} mA/nm\n")
    return modulation_index

def power_spectrum_density(data_array, N, dt):
    fft_amplitude = fft(data_array)[:N//2] # fourier transform of the current density
    fft_freq = fftfreq(N, dt)[:N//2] # fourier frequencies
    array_PSD = ((np.abs(fft_amplitude))**2)*2//N # type: ignore # power spectral density, normalized (single sided,  N datapoints)
    return array_PSD, fft_freq

def plot_fourier(df, ax, logy=False):
    df.plot(
        ax=ax,
        x='Frequency',
        y=['Amplitude','Smoothed amplitude'],
        title='Fourier transform of current density in Gunn diode',
        xlabel='Frequency (THz)',
        ylabel='Current PSD',
        xlim=([0, plot_focus]) if logy else None,
        logy=logy,
        **plot_config
    )

plot_config = {
    'figsize': figure_size,
    'style': ['b:','r-'],
    'grid': True
}
# =================================================================================================

# M A I N   C O D E

print("\nSINGLE DATA FILE ANALYSIS\n")
print("CURRENT DENSITY OSCILLATIONS IN GUNN DIODE\n")
print("------------------------------------------\n")
print("Reading and processing data file...\n")

set_path() # set the path to the current file location
setup_names(file_id) # set the file names

dataframe = pd.read_csv(datafile_name, names=['Current'])
dataframe['Time'] = dataframe.index*dt
dataframe = dataframe[['Time','Current']]

data = smoothening(dataframe,window_size)
data = data[data.index >= 200000] # ignore thermalization data

print(f"File {datafile_name} processed successfully. Showing some statistics and some rows:\n")
print(data.describe())  # print some statistics
print(data)  # print the datafile

mod_index = modulation_index(data)

# plot the datafile
print("\nPlotting the data...")
first_plot, axs = plt.subplots(2, figsize=figure_size) # create a figure with 1 subplot

N=len(data) # number of data points
print(f"Number of data points: {N} ")
print(f"Time step: {dt} fs")

currents_array=data['Current'].to_numpy()
smoothened_array=data['Smoothened_Current'].to_numpy()
dc_current = np.mean(currents_array)
dc_smoothened = np.mean(smoothened_array)
ac_array = currents_array - dc_current
smoothened_ac_array = smoothened_array - dc_smoothened

print(f"Mean current density: {round(dc_current)} mA/nm")

ac_df=pd.DataFrame({
    'Time':data['Time'],
    'AC':ac_array,
    'Smoothened_AC':smoothened_ac_array
}) # type: ignore # create a dataframe with the oscillation data

first_plot = data.plot(
    ax=axs[0], # plot the data in the first subplot
    x='Time',
    y=['Current','Smoothened_Current'],
    title='Current density in Gunn diode',
    xlabel='Time (fs)',
    ylabel=' Drain current density',
    xlim=([2e5,5e5]),  # limit x axis
    **plot_config
)
first_plot = ac_df.plot(
    ax=axs[1], # plot the data in the second subplot
    x='Time',
    y=['AC','Smoothened_AC'],
    title='Current density in Gunn diode, without offset',
    xlabel='Time (fs)',
    ylabel='Drain current density',
    xlim=([2e5,5e5]),  # limit x axis
    **plot_config
)
plt.tight_layout()
plt.savefig(plot_name) # save the plot as a png file

print(f"Plot saved as {plot_name}\n")

# FOURIER TRANSFORM

if perform_fourier:
    print("\nPerforming Fourier transform...\n")

    ac_amplitude, ac_frequency = power_spectrum_density(ac_array, N, dt)
    smoothened_amplitude, _ = power_spectrum_density(smoothened_ac_array, N, dt)

    PSD_df = pd.DataFrame({
    'Frequency': ac_frequency, 
    'Amplitude': ac_amplitude,
    'Smoothed amplitude': smoothened_amplitude
    })

    print("Fourier transform performed successfully. Showing some statistics and some rows:\n")

    print("\n", PSD_df.describe())  # print some statistics
    print("\n", PSD_df)  # print the fourier data

    print("\nPlotting the Fourier transform...")

    fourier_plot, axs = plt.subplots(2, figsize=figure_size)
    plot_fourier(PSD_df, axs[0])
    plot_fourier(PSD_df, axs[1], logy=True)
    plt.tight_layout()
    plt.savefig(fourier_name) # save the fourier plot as a png file
    print(f"Fourier plot saved as {fourier_name}\n")

timeit = time.time() - start_time
print("\nAnalysis completed in ", round(timeit, 3), " seconds.\n")