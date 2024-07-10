import os # for file management
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib # for plotting
import matplotlib.pyplot as plt # for plotting
matplotlib.use('Agg') # for non-interactive plotting
from tqdm import tqdm # for progress bar

# PARAMETERS ======================================================================================
parent_dir = 'D:/datos_montecarlo'
datafile_name = 'corr_reduced'
plot_name = 'plot.png'
exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/exit.csv'
dt=2e-16 # 0,2 fs time step
window_size = 2000 # window size for smoothing
perform_plots = True
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
    modulation_index = (max_current - min_current) / ( data['Smoothened_Current'].mean())
    return modulation_index

plot_config = {
    'figsize': (8,6),
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

    if perform_plots:
        # plot the datafile
        data.plot(
            x='Time',
            y=['Current','Smoothened_Current'],
            title='Current density in Gunn diode',
            xlabel='Time (s)',
            ylabel=' Drain current density ($A/m$)',
            xlim=([2e5*dt,5e5*dt]),  # limit x axis
            **plot_config
        )
        plt.margins(x=0)
        plt.tight_layout()

        # Textbox content using Matplotlib's MathText
        textstr = '\n'.join((
            r'$W_0$ = ' + f'{Wo:.0f} nm',
            r'$V_{DS}$ = ' + f'{Vds:.0f} V',
            r'T = ' + f'{Temp:.0f} K',
            r'$\epsilon_{1 \rightarrow 2}$ = ' + intervalley + ' eV',
            r'$N_D$ = ' + f'{Nd}' + r' $m^{-2}$',
            r'Mod. index = ' + f'{mod_index:.4f}'
            ))

        # Place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

        dir=os.path.dirname(datafile_name)
        plot_path=os.path.join(dir,plot_name)
        plt.savefig(plot_path) # save the plot as a png file
        plt.close('all')
 
    exit_df = pd.DataFrame({
        'Wo': [Wo],
        'Vds': [Vds],
        'Temp': [Temp],
        'Intervalley': [intervalley],
        'Nd': [Nd],
        'Mod index': [mod_index]
    })

    header = not os.path.isfile(exit_path)
    exit_df.to_csv(exit_path, mode='a', header=header, index=False)

for file in tqdm(file_paths):
    process_file(file)