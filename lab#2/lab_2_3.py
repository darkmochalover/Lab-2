import numpy as np
import csv
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

BMI_label_dict = {''}

# read csv file
df = pd.read_csv('data/bmi_data_lab2.csv')
print(df)

'''
Lab 2-3
Missing value manipulation (simple)

* Identify all dirty records with likely-wrong or missing height or weight values
* Remove all likely-wrong value. (Replace with NAN)
* print # of rows with NAN, and # of NAN for each column.
* extract all rows without NAN
* Fill NAN with ean, median, or using ffill/ bfill methods
'''
print("\n\n[Original Data]")
print(df)

### * Remove all likely-wrong value. (Replace with NAN)
# remove likely-wrong values
'''
140cm ~ 190cm  . . . About 56 Inches ~ 75 Inches
40kg ~ 120kg . . . About 88 Pounds ~ 264 Pounds
'''
# Replace all values outside the valid range with NaN
df.loc[(df['Height (Inches)'] < 56) | (df['Height (Inches)'] >= 75), 'Height (Inches)'] = np.nan
df.loc[(df['Weight (Pounds)'] < 88) | (df['Weight (Pounds)'] >= 264), 'Weight (Pounds)'] = np.nan

print("\n\n[Replace all values outside the valid range with NaN]")
print(df)

### * print # of rows with NAN, and # of NAN for each column.
# Print the number of rows with NaN
print("Number of rows with NaN:", df.isnull().any(axis=1).sum())

# Print the number of NaN for each column
print("Number of NaN for each column:")
print(df.isnull().sum())

### Extract all rows without NaN
df_clean = df.dropna()
print(df_clean)


###
## Fill NaN with mean:
df_mean = df.fillna(df.mean())
print("\n\n[DataFrame with NaN filled with mean]")
print(df_mean)  

## Fill NaN with median:
df_median = df.fillna(df.median())
print("\n\n[DataFrame with NaN filled with median]")
print(df_median)    

## Fill NaN with forward fill
df_ffill = df.fillna(method='ffill')
print("\n\n[DataFrame with NaN filled with forward fill]") 
print(df_ffill)

## Fill NaN with backward fill
df_bfill = df.fillna(method='bfill')
print("\n\n[DataFrame with NaN filled with backward fill]")
print(df_bfill)


