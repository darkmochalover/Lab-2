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
Peek into the dataset (data exploation)
(2/7)
'''

# Print statistical data
print(df.describe())

# print feature names
print(df.columns)

# print data type
print(df.dtypes)


# plot weight histogram for each BMI value
fig = plt.figure(figsize=(10,8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(df[df.BMI == n]['Weight (Pounds)'], bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Weight Histogram")
plt.show()
# plt.savefig('lab#2/graph/Weight Histogram.png', format='png')

# plot height histogram for each BMI value
plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(df[df.BMI == n]['Height (Inches)'], bins=10, label="{}".format(n))
    plt.xlabel('Height')
    plt.ylabel('Frequency')

plt.suptitle("Height Histogram")
plt.show()
# plt.savefig('lab#2/graph/Height Histogram.png', format='png')

### Height
## Fit and transform the height values using each scaler instance
scaled_height__Standard = StandardScaler().fit_transform(df[['Height (Inches)', 'BMI']])
scaled_height__MinMax = MinMaxScaler().fit_transform(df[['Height (Inches)', 'BMI']])
scaled_height__Robust = RobustScaler().fit_transform(df[['Height (Inches)', 'BMI']])

plt.figure(figsize=(6,12))      # Set figure

## Create histograms for each scaled height value
# StandardScaler
plt.subplot(3, 1, 1)
plt.hist(scaled_height__Standard, bins=10, alpha=0.5, label='StandardScaler')
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('StandardScaler')

# MinMaxScaler
plt.subplot(3, 1, 2)
plt.hist(scaled_height__MinMax, bins=10, alpha=0.5, label='MinMaxScaler')
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('MinMaxScaler')

# RobustScaler
plt.subplot(3, 1, 3)
plt.hist(scaled_height__Robust, bins=10, alpha=0.5, label='RobustScaler')
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('RobustScaler')

# Add labels and legend
plt.suptitle('Scaling Histograms for Height')
plt.legend(loc='upper right')

# Show the plot
plt.show()
plt.savefig('lab#2/graph/scaled_height_value.png', format='png')

### Weight
## Fit and transform the weight values using each scaler instance
scaled_weight__Standard = StandardScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])
scaled_weight__MinMax = MinMaxScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])
scaled_weight__Robust = RobustScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])

plt.figure(figsize=(6,12))     # Set figure

## Create histograms for each scaled weight value
# StandardScaler
plt.subplot(3, 1, 1)
plt.hist(scaled_weight__Standard, bins=10, alpha=0.5, label='StandardScaler')
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('StandardScaler')


# MinMaxScaler
plt.subplot(3, 1, 2)
plt.hist(scaled_weight__MinMax, bins=10, alpha=0.5, label='MinMaxScaler')
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('MinMaxScaler')

# RobustScaler
plt.subplot(3, 1, 3)
plt.hist(scaled_weight__Robust, bins=10, alpha=0.5, label='RobustScaler')
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('RobustScaler')


# Add labels and legend
plt.suptitle('Scaling Histograms for Height')
plt.legend(loc='upper right')

# Show the plot
plt.show()
plt.savefig('lab#2/graph/scaled_weight_value.png', format='png')