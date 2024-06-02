import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parent_dir = 'D:/datos_montecarlo'
heatmaps_dir = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/heatmaps'

df = pd.read_csv('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/exit.csv')

unique_temperatures = [300.0,400.0,500.0]
unique_impurifications = [5e+23,1e+24,5e+24,1e+25]

for temp in unique_temperatures:
    for Nd in unique_impurifications:
        
        # Filter dataframe for each temperature and impuritiy
        filter_df = df[np.isclose(df['Temp'], temp) & np.isclose(df['Nd'], Nd)]
        
        # Pivot the DataFrame (arrange like a matrix)
        mod_df = filter_df.pivot(index='Wo', columns='Vds', values='Mod index')
        freq_df = filter_df.pivot(index='Wo', columns='Vds', values='FMA')

        # Convert the columns to integers
        mod_df.columns = mod_df.columns.astype(int)
        freq_df.columns = freq_df.columns.astype(int)
        mod_df.index = mod_df.index.astype(int)
        freq_df.index = freq_df.index.astype(int)

        # Invert the order of the index
        mod_df = mod_df.sort_index(ascending=False)
        freq_df = freq_df.sort_index(ascending=False)

        print(f'Temperature: {temp} K, Impurification: {Nd} m^-3')
        print(mod_df)
        print(freq_df)
        print('=======================================================================================\n')
        
        # Create the subplot
        fig, axs = plt.subplots(1, 2, figsize=(24, 8))

        sns.heatmap(mod_df, ax=axs[0], cbar_kws={'label': 'Mod. index'})
        axs[0].set_title('Mod index')
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)  # Rotate x-axis labels

        sns.heatmap(freq_df, ax=axs[1], cbar_kws={'label': 'FMA (Hz)'})
        axs[1].set_title('Main F Freq')
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)  # Rotate x-axis labels

        # Add a title to the entire figure
        fig.suptitle(f'Temperature: {temp} K, Impurification: {Nd} m^-3')

        # Save the heatmap
        plt.tight_layout()
        plt.savefig(os.path.join(heatmaps_dir, f'heatmap_{temp}(K)_{Nd}(m-3).png'))
        plt.clf()

        del filter_df, mod_df, freq_df