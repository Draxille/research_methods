import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read csv file into dataframe
pandas_df_full = pd.read_csv("Results_21MAR2022_nokcaladjust.csv")

# One-hot encode the categorical variables
categorical_cols = ['age_group', 'diet_group', 'sex']
data_encoded = pd.get_dummies(pandas_df_full, columns=categorical_cols)

# Select only the columns for environmental factors and the new dummy variables
env_columns = [col for col in pandas_df_full.columns if 'mean_' in col or 'sd_' in col]  # adjust if necessary
encoded_columns = [col for col in data_encoded.columns if col.startswith(tuple(categorical_cols))]
selected_columns = env_columns + encoded_columns

# Calculate the correlation matrix for the selected columns
corr_matrix = data_encoded[selected_columns].corr()

# Plot the heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix with Categorical and Environmental Factors')
plt.show()