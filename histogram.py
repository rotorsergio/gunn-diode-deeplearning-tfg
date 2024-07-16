import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read data from CSV
exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/exit.csv'
dataframe = pd.read_csv(exit_path)

# Step 3: Create the histogram
sns.histplot(dataframe['Mod index'], bins=30, color='blue')

# Optional: Customize your plot (e.g., add a title)
plt.title('Histogram of Data Column')

# Step 4: Display the plot
plt.show()