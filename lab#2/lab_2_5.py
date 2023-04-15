import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression

BMI_label_dict = {''}

# read csv file
df = pd.read_csv('data/bmi_data_lab2.csv')
print(df)

'''
Lab 2-5
Missing value manipulation (more elaborate)

* Identify all dirty records with likely-wrong or missing height or weight values
* Clean the dirty values using linear regression
* Draw a scatter plot of height and weight, in the clean dataset 
emphasizing previously dirty records with a different color
'''
print("\n\n[Original Data]")
print(df)

# remove likely-wrong values
'''
140cm ~ 190cm  . . . About 56 Inches ~ 75 Inches
40kg ~ 120kg . . . About 88 Pounds ~ 264 Pounds
'''
df_dirty = df
# Replace all values outside the valid range with NaN
df_dirty.loc[(df_dirty['Height (Inches)'] < 56) | (df_dirty['Height (Inches)'] >= 75), 'Height (Inches)'] = np.nan
df_dirty.loc[(df_dirty['Weight (Pounds)'] < 88) | (df_dirty['Weight (Pounds)'] >= 264), 'Weight (Pounds)'] = np.nan

print("\n\n[Replace all values outside the valid range with NaN]")
print(df_dirty)


# Split the DataFrame into two parts: one with the dirty values and one without
df_clean = df_dirty.dropna()
df_dirty = df_dirty[df_dirty.isnull().any(axis=1)]
print('\n[df_dirty]\n', df_dirty)

'''
# (2-6) 
Fit a linear regression model to the part without dirty values
'''
# Input feature: 'Weight (Pounds)'
# Output feature: 'Height (Inches)'
height_pred_model = LinearRegression()
height_pred_model.fit(df_clean[['Weight (Pounds)']], df_clean[['Height (Inches)']])

# Input feature: 'Height (Inches)'
# Output feature: 'Weight (Pounds)'
weight_pred_model = LinearRegression()
weight_pred_model.fit(df_clean[['Height (Inches)']], df_clean[['Weight (Pounds)']])

# Input feature: 'Height (Inches)', 'Weight (Pounds)'
# Output feature: 'BMI'
bmi_pred_model = LinearRegression()
bmi_pred_model.fit(df_clean[['Height (Inches)', 'Weight (Pounds)']], df_clean[['BMI']])

# Use the fitted model to predict the missing values in the part with dirty values

#Check Null Value First!
print(pd.isnull(df_dirty).sum())

## Height Prediection (Weight -> Height)
h_data = df_dirty.dropna(subset=['Weight (Pounds)'], how='any', axis=0)  # load non-NaN value 

print("[height_pred]")
h_data['Height (Inches)'] = height_pred_model.predict(h_data[['Weight (Pounds)']]) # predict with model
print(h_data)

print("[Update Predicted Height Value]")
df_dirty.update(h_data, overwrite=False)
print(df_dirty)


## Weight Prediection (Weight -> Height)
w_data = df_dirty.dropna(subset=['Height (Inches)'], how='any', axis=0) # load non-NaN value 
print("[weight_pred]")
w_data['Weight (Pounds)'] = weight_pred_model.predict(w_data[['Height (Inches)']]) # predict with model
print(w_data)

print("[Update Predicted Weight Value]")
df_dirty.update(w_data, overwrite=False)
print(df_dirty)


## BMI Prediection (Weight + Height -> BMI)
bmi_data = df_dirty.dropna(subset=['BMI'], how='any', axis=0) # load non-NaN value 
print("[bmi_pred]")
bmi_data['BMI'] = bmi_pred_model.predict(bmi_data[['Height (Inches)', 'Weight (Pounds)']]) # predict with model
print(bmi_data)

print("[Update Predicted Weight Value]")
df_dirty.update(bmi_data, overwrite=False)
print(df_dirty)


# Combine the two parts back into a single DataFrame
df_clean = pd.concat([df_clean, df_dirty])

# Print the cleaned DataFrame
print(df_clean)


'''
* Draw a scatter plot of height and weight, in the clean dataset 
emphasizing previously dirty records with a different color
'''
# Create a scatter plot of height and weight in the clean dataset
plt.scatter(df_clean['Height (Inches)'], df_clean['Weight (Pounds)'], color='blue')

# Emphasize previously dirty records with a different color
plt.scatter(df_dirty['Height (Inches)'], df_dirty['Weight (Pounds)'], color='red')

# Add labels and title to the plot
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Height and Weight')

# Show the plot
plt.show()