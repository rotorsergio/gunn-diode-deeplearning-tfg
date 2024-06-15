import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

heatmaps_dir = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/heatmaps'
datasets_dir = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets'

datamode='std_prediction' # Modes: 'original', 'norm_prediction', 'std_prediction'
map_modes = ['wo-v', 'nd-v']
mode = map_modes[0]

if datamode == 'original':
    df = pd.read_csv(os.path.join(datasets_dir, 'exit.csv'))
    save_dir = os.path.join(heatmaps_dir, 'original')
elif datamode == 'norm_prediction':
    df = pd.read_csv(os.path.join(datasets_dir, datamode + '.csv'))
    save_dir = os.path.join(heatmaps_dir, 'norm_prediction')
elif datamode == 'std_prediction':
    df = pd.read_csv(os.path.join(datasets_dir, datamode + '.csv'))
    save_dir = os.path.join(heatmaps_dir, 'std_prediction')
else:
    raise ValueError('Invalid data mode')

df['Wo']=df['Wo'].astype(int)
df['Vds']=df['Vds'].astype(int)
df['Temp']=df['Temp'].astype(int)
df['Nd'] = df['Nd'].apply(lambda x: round(x, -int(np.floor(np.log10(abs(x)))))) # Round to the nearest order of magnitude

global_min = df['Mod index'].min()
global_max = df['Mod index'].max()

# Dynamically obtain unique values of the fixed variable
unique_temperatures = sorted(pd.unique(df['Temp']))
unique_impurifications = sorted(pd.unique(df['Nd']))
unique_w0 = sorted(pd.unique(df['Wo']))

if mode == map_modes[0]:
    fixed_vars = unique_impurifications
elif mode == map_modes[1]:
    fixed_vars = unique_w0
else:
    raise ValueError('Invalid heatmap mode')

# Outer progress bar
with tqdm(total=len(fixed_vars), desc="Overall Progress", position=0) as pbar1:
    for var in fixed_vars:
        fig, axs = plt.subplots(1, len(unique_temperatures), figsize=(16,9))
        
        # Inner progress bar
        with tqdm(total=len(unique_temperatures), desc="Inner Progress", position=1, leave=False) as pbar2:
            for i, temp in enumerate(unique_temperatures):

                if mode == map_modes[0]:
                    filter_df = df[(df['Temp'] == temp) & (df['Nd'] == var)]
                    mod_df = filter_df.pivot(index='Wo', columns='Vds', values='Mod index')
                    mod_df = mod_df.sort_index(ascending=False)
                    fixed_var_text = f'$Nd: {var} m^{{-3}}$'
                elif mode == map_modes[1]:
                    filter_df = df[(df['Temp'] == temp) & (df['Wo'] == var)]
                    mod_df = filter_df.pivot(index='Nd', columns='Vds', values='Mod index')
                    mod_df = mod_df.sort_index(ascending=False)
                    fixed_var_text = f'$Wo: {var} nm$'

                im = sns.heatmap(mod_df, ax=axs[i], annot=False, cmap='coolwarm', vmin=global_min, vmax=global_max)
                axs[i].set_title(f'Temperature: {temp} K')

                axs[i].text(0.5, 0.5, f'{fixed_var_text}\n$\\varepsilon_{{1,2}} = 0.9 \\times 10^{{12}}$', 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=axs[i].transAxes, color='white', fontsize=14)

                if mode == map_modes[0]:
                    plt.savefig(os.path.join(heatmaps_dir, f'{datamode}', f'{mode}', f'{var}.png'))
                elif mode == map_modes[1]:
                    plt.savefig(os.path.join(heatmaps_dir, f'{datamode}', f'{mode}', f'{var}.png'))

                pbar2.update()  # Update inner progress bar after each temperature

        pbar1.update()  # Update outer progress bar after each variable