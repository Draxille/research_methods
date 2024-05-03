#Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import to_rgba

#Read csv file into dataframe
pandas_df_full = pd.read_csv("Results_21MAR2022_nokcaladjust.csv")

# Select relevant environmental factors
df_selected = pandas_df_full[['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
                              'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid',
                              'diet_group', 'sex']]

# Create a column for diet group-sex combinations
df_selected['diet_sex_combination'] = df_selected['diet_group'] + '_' + df_selected['sex']

# Normalize the environmental factors using Min-Max scaling
scaler = MinMaxScaler()
X = df_selected[['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
                 'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid']]
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine the normalized environmental factors with the diet group and sex data
df_normalized = pd.concat([X_normalized, df_selected[['diet_group', 'sex', 'diet_sex_combination']]], axis=1)

# Set the color palette for diet groups
diet_sex_combinations = df_selected['diet_sex_combination'].unique()
palette = {}
for combination in diet_sex_combinations:
    if 'vegan' in combination:
        if 'female' in combination:
            palette[combination] = 'blue'
        else: 
            palette[combination] = (to_rgba('blue')[0]*0.5,to_rgba('blue')[1]*0.5,to_rgba('blue')[2]*0.5,to_rgba('blue')[3])
    elif 'veggie' in combination:
        if 'female' in combination:
            palette[combination] = 'green'
        else:
            palette[combination] = (to_rgba('green')[0]*0.5,to_rgba('green')[1]*0.5,to_rgba('green')[2]*0.5,to_rgba('green')[3])
    elif 'meat100' in combination:
        if 'female' in combination:
            palette[combination] = 'orange'
        else:
            palette[combination] = (to_rgba('orange')[0]*0.8,to_rgba('orange')[1]*0.8,to_rgba('orange')[2]*0.8,to_rgba('orange')[3])
    elif 'meat50' in combination:
        if 'female' in combination:
            palette[combination] = 'purple'
        else:
            palette[combination] = (to_rgba('purple')[0]*0.6,to_rgba('purple')[1]*0.6,to_rgba('purple')[2]*0.6,to_rgba('purple')[3])
    elif 'meat' in combination:
        if 'female' in combination:
            palette[combination] = 'red'
        else:
            palette[combination] = (to_rgba('red')[0]*0.7,to_rgba('red')[1]*0.7,to_rgba('red')[2]*0.7,to_rgba('red')[3])
    elif 'fish' in combination:
        if 'female' in combination:
            palette[combination] = 'cyan'
        else:
            palette[combination] = (to_rgba('cyan')[0]*0.6,to_rgba('cyan')[1]*0.6,to_rgba('cyan')[2]*0.6,to_rgba('cyan')[3])
    else:
        palette[combination] = 'black'  # A fallback color

# Create a markers dictionary for all combinations of diet and sex
markers = {combination: ('o' if 'female' in combination else 'P') for combination in diet_sex_combinations}

# Plot the scatterplot matrix
sns.pairplot(df_normalized, hue='diet_sex_combination', palette=palette, markers=markers)
plt.show()