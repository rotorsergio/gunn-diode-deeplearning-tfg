import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

heatmaps_dir = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/heatmaps'
datasets_dir = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets'

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

clear_directory(heatmaps_dir)

datamode='fine_prediction' # Modes: 'original', 'og_prediction', 'norm_prediction', 'fine, 'fine_prediction'
map_modes = ['wo-v', 'nd-v']
mode = map_modes[1]

match datamode:
    case 'original':
        df = pd.read_csv(os.path.join(datasets_dir, 'exit.csv'))
        save_dir = os.path.join(heatmaps_dir, 'original')
    case 'og_prediction':
        df = pd.read_csv(os.path.join(datasets_dir, datamode + '.csv'))
        save_dir = os.path.join(heatmaps_dir, 'og_prediction')
    case 'norm_prediction':
        df = pd.read_csv(os.path.join(datasets_dir, datamode + '.csv'))
        save_dir = os.path.join(heatmaps_dir, 'norm_prediction')
    case 'fine':
        df = pd.read_csv(os.path.join(datasets_dir, '200nm.csv'))
        save_dir = os.path.join(heatmaps_dir, 'fine')
    case 'fine_prediction':
        df = pd.read_csv(os.path.join(datasets_dir, datamode + '.csv'))
        save_dir = os.path.join(heatmaps_dir, 'fine_prediction')
    case _:
        raise ValueError('Invalid data mode')

df['Wo']=df['Wo'].astype(int)
df['Vds']=df['Vds'].astype(int)
df['Temp']=df['Temp'].astype(int)
df['Nd'] = df['Nd'].apply(lambda x: round(x, 1-int(np.floor(np.log10(abs(x)))))) # Round to one decimal place past the nearest order of magnitude

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
        fig, axs = plt.subplots(1, len(unique_temperatures), figsize=(14,5))
        fixed_var_text = ""  # Initialize outside the loop

        # Adjust subplot parameters to make room for the text box at the bottom
        plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin

        # Inner progress bar
        with tqdm(total=len(unique_temperatures), desc="Inner Progress", position=1, leave=False) as pbar2:
            for i, temp in enumerate(unique_temperatures):

                if mode == map_modes[0]:
                    filter_df = df[(df['Temp'] == temp) & (df['Nd'] == var)]
                    mod_df = filter_df.pivot(index='Wo', columns='Vds', values='Mod index')
                    mod_df = mod_df.sort_index(ascending=False)
                    fixed_var_text = rf'$N_D = {var} \  m^{{-2}}$'
                elif mode == map_modes[1]:
                    filter_df = df[(df['Temp'] == temp) & (df['Wo'] == var)]
                    mod_df = filter_df.pivot(index='Nd', columns='Vds', values='Mod index')
                    mod_df = mod_df.sort_index(ascending=False)
                    fixed_var_text = rf'$W_O = {var} nm$'

                im = sns.heatmap(mod_df, ax=axs[i], annot=False, cmap='coolwarm')
                axs[i].set_title(f'Temperature = {temp} K')

                pbar2.update()  # Update inner progress bar after each temperature
        
        plt.tight_layout(rect=(0, 0.1, 1, 1))

        # Add the text box outside the inner loop
        if mode == map_modes[0] or mode == map_modes[1]:
            fig.text(0.5, 0.05,
                     f'{fixed_var_text}     $\\varepsilon_{{1,2}} = 0.9 \\times 10^{{12}}$ eV',
                     ha='center',
                     va='center',
                     color='black', fontsize=14,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7)
                     )

        # Save the figure after adding the text box
        plt.savefig(os.path.join(heatmaps_dir, f'{datamode}', f'{mode}', f'{var}.png'))

        plt.close()
        pbar1.update()  # Update outer progress bar after each variable