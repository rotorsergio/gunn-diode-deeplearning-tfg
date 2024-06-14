import time
import os # for file management
import glob # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib # for plotting
import matplotlib.pyplot as plt # for plotting
matplotlib.use('Agg') # for non-interactive plotting
from scipy.fft import fft,fftfreq # scientific computing (used for fourier transform)
from tqdm import tqdm # for progress bar
import concurrent.futures # for parallel processing

start_time = time.time()

# PARAMETERS ======================================================================================
parent_dir = 'D:/datos_montecarlo'
datafile_name = 'corr_reduced'
plot_name = 'plot.png'
fourier_name = 'fourier.png'
partial_exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/partial_exits'
exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/exit.csv'
partial_files = glob.glob(os.path.join(partial_exit_path, "exit_*.csv"))
dt=2e-16 # 0,2 fs time step
window_size = 2000 # window size for smoothing
plot_focus=2e14 # focus on the first 10 GHz, set to .5 for max range
figure_size=(16,9) # figure size
perform_fourier = True
perform_plots = True
perform_multiprocessing = False
# =================================================================================================

# DEFINED FUNCTIONS ===============================================================================
def obtain_paths():
    files = [os.path.join(dirpath, file)
                for dirpath, dirnames, filenames in os.walk(parent_dir)
                for file in filenames if file == datafile_name]
    return files
def read_file(file_path):
    dataframe = pd.read_csv(file_path, names=['Current'])
    dataframe['Time'] = dataframe.index*dt
    dataframe = dataframe[['Time','Current']]
    return dataframe
def smoothening(valid_data):
    valid_data['Smoothened_Current'] = valid_data['Current'].rolling(window=window_size).mean()
    valid_data = valid_data.dropna()
    return valid_data
def modulation_index(data):
    max_current = data['Smoothened_Current'].max() # maximum current density
    min_current = data['Smoothened_Current'].min() # minimum current density
    modulation_index = (max_current - min_current) / ( data['Smoothened_Current'].mean()) # modulation index
    return modulation_index
def power_spectrum_density(data_array, N):
    fft_amplitude = fft(data_array)[:N//2] # fourier transform of the current density
    fft_freq = fftfreq(N, dt)[:N//2] # fourier frequencies
    array_PSD = ((np.abs(fft_amplitude))**2)*2//N # type: ignore # power spectral density, normalized (single sided, N datapoints)
    return array_PSD, fft_freq
def plot_fourier(df, ax, logy=False):
    df.plot(
        ax=ax,
        x='Frequency',
        y=['Amplitude','Smoothed amplitude'],
        title='Fourier transform of current density in Gunn diode',
        xlabel='Frequency (Hz)',
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

print("\nCOMPLETE DATASET ANALYSIS\n")
print("CURRENT DENSITY OSCILLATIONS IN GUNN DIODE\n")
print("==================================================\n")

file_paths = obtain_paths() # Obtain the paths of all the files in the parent directory

if os.path.isfile(exit_path):
    os.remove(exit_path)

def process_file(datafile_name):

    rel_path = os.path.relpath(datafile_name, parent_dir)
    path_components = rel_path.split(os.sep)

    # Extract the parameters from the path
    Wo = float(path_components[0][-3:])
    Vds = float(path_components[1][-4:])
    Temp = float(path_components[2][-5:])
    exp_part, dec_part = path_components[3].split('_')
    exp_part = exp_part[1:]
    intervalley = format(float(f"{dec_part}e{exp_part}"),'.1e')
    Nd = float(path_components[4][-6:])
    
    dataframe = read_file(datafile_name)
    data = smoothening(dataframe)
    data = data[data.index >= 200000] # ignore thermalization data
    mod_index = modulation_index(data)

    currents_array=data['Current'].to_numpy()
    smoothened_array=data['Smoothened_Current'].to_numpy()
    dc_current = np.mean(currents_array)
    dc_smoothened = np.mean(smoothened_array)
    ac_array = currents_array - dc_current
    smoothened_ac_array = smoothened_array - dc_smoothened

    ac_df=pd.DataFrame({
        'Time':data['Time'],
        'AC':ac_array,
        'Smoothened_AC':smoothened_ac_array
    }) # type: ignore # create a dataframe with the oscillation data

    if perform_plots:
        # plot the datafile
        data.plot(
            x='Time',
            y=['Current','Smoothened_Current'],
            title='Current density in Gunn diode',
            xlabel='Time (s)',
            ylabel=' Drain current density',
            xlim=([2e5*dt,5e5*dt]),  # limit x axis
            **plot_config
        )
        plt.tight_layout()

        dir=os.path.dirname(datafile_name)
        plot_path=os.path.join(dir,plot_name)
        plt.savefig(plot_path) # save the plot as a png file
        plt.close('all')

    # FOURIER TRANSFORM

    if perform_fourier:
        N=len(data) # number of data points

        windowed_ac = np.hanning(N)*ac_array # apply a hanning window to the data
        windowed_smoothened = np.hanning(N)*smoothened_ac_array # apply a hanning window to the data

        ac_amplitude, ac_frequency = power_spectrum_density(windowed_ac, N)
        smoothened_amplitude, _ = power_spectrum_density(windowed_smoothened, N)

        PSD_df = pd.DataFrame({
            'Frequency': ac_frequency, 
            'Amplitude': ac_amplitude,
            'Smoothed amplitude': smoothened_amplitude
        })

        FMA = PSD_df['Frequency'][PSD_df['Amplitude'].idxmax()] # Frequency of Maximum Amplitude  

        if perform_plots:
            _, axs = plt.subplots(2, figsize=figure_size)
            plot_fourier(PSD_df, axs[0])
            plot_fourier(PSD_df, axs[1], logy=True)
            plt.tight_layout()
            fourier_path=os.path.join(dir,fourier_name)
            plt.savefig(fourier_path) # save the fourier plot as a png file
            
            plt.close('all')

        exit_df = pd.DataFrame({
            'Wo': [Wo],
            'Vds': [Vds],
            'Temp': [Temp],
            'Intervalley': [intervalley],
            'Nd': [Nd],
            'Mod index': [mod_index],
            'FMA': [FMA]
        })
    else:
        FMA = 0
        exit_df = pd.DataFrame({
            'Wo': [Wo],
            'Vds': [Vds],
            'Temp': [Temp],
            'Intervalley': [intervalley],
            'Nd': [Nd],
            'Mod index': [mod_index],
            'FMA': [FMA] # FMA = Frequency of Maximum Amplitude
        })

    header = not os.path.isfile(partial_exit_path)
    exit_df.to_csv(os.path.join(partial_exit_path, f"exit_{os.getpid()}.csv"), mode='a', header=header, index=False)

if perform_multiprocessing:
    if __name__ == '__main__':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
            executor.shutdown(wait=True)  # Wait for all tasks to complete
    
    # Combine all the files into a single file

    df_from_each_file = (pd.read_csv(f) for f in partial_files)
    combined_df = pd.concat(df_from_each_file, ignore_index=True)
    combined_df.to_csv(exit_path, index=False)
    for file in partial_files:
        os.remove(file)

else:
    for file in tqdm(file_paths):
        header = not os.path.isfile(os.path.join(partial_exit_path, f"exit_{os.getpid()}.csv"))
        process_file(file)
    
    # Move and rename the file
    os.rename(os.path.join(partial_exit_path, f"exit_{os.getpid()}.csv"), exit_path)

timeit = time.time() - start_time