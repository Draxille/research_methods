#Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Read csv file into dataframe
pandas_df_full = pd.read_csv("Results_21MAR2022_nokcaladjust.csv")

#Create a list of relevant environmental factors
selected_columns = [
    'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
    'mean_ghgs_ch4', 'diet_group'
]

# Create a new DataFrame with the selected columns
df_selected = pandas_df_full[selected_columns]

# Separate the environmental metrics and diet group column
X = df_selected.drop(columns=['diet_group'])
diet_group = df_selected['diet_group']

# Normalize the environmental factors using Min-Max scaling
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine the normalized environmental factors with the diet group column
df_normalized = pd.concat([X_normalized, diet_group], axis=1)

# Set the color palette for diet groups
palette = {'vegan': 'blue', 'veggie': 'green', 'meat': 'red', 'meat50': 'purple', 'meat100': 'orange', 'fish': 'cyan'}

# Plot the scatterplot matrix
sns.pairplot(df_normalized, hue='diet_group', palette=palette, markers=["o", "s", "D", "^", "v", "P"])
plt.show()