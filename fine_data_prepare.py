import pandas as pd
import numpy as np
import os

# ===== READ FWF AND CREATE CSV =====
df = pd.read_csv('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/TABLON_TOTAL', sep='\t', header=0)

# Filter the columns that are not needed
df = df[['vds', 'temperatura', 'nd', 'imodulacion']]
df['nd'] = df['nd'].apply(lambda x: x.replace(',', '.')).astype(float) # Replace ',' by '.' and convert to float
df['imodulacion'] = df['imodulacion'].apply(lambda x: x.replace(',', '.')).astype(float) # Replace ',' by '.' and convert to float
df.columns = ['Vds', 'Temp', 'Nd', 'Mod index'] # Rename the columns
df['Wo'] = 200 # Add the column 'Wo' with value 200
df=df[['Wo', 'Vds', 'Temp', 'Nd', 'Mod index']] # Reorder the columns. Now it has the same structure as all other datasets

#Erase columns with Temp > 500
df = df[df['Temp'] <= 500]

df.to_csv('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/200nm.csv', index=False)