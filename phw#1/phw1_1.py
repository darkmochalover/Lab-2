import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression

# read excel file
df = pd.read_excel('data/bmi_data_phw1.xlsx')
print(df)

'''
PHW1-1
Data Exploration
* Print dataset statistical data, feature names & data types
* Plot height & weight histograms (bins=10) for each BMI value
* Plot scaling results for height and weight
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

# plot height histogram for each BMI value
plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(df[df.BMI == n]['Height (Inches)'], bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Height Histogram")
plt.show()

### Height
## Fit and transform the height values using each scaler instance
scaled_height__Standard = StandardScaler().fit_transform(df[['Height (Inches)', 'BMI']])
scaled_height__MinMax = MinMaxScaler().fit_transform(df[['Height (Inches)', 'BMI']])
scaled_height__Robust = RobustScaler().fit_transform(df[['Height (Inches)', 'BMI']])

plt.figure(figsize=(11,5))     # Set figure

## Create histograms for each scaled height value
# StandardScaler
plt.subplot(1, 3, 1)
plt.hist(scaled_height__Standard, bins=10, alpha=0.5)
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('StandardScaler')

# MinMaxScaler
plt.subplot(1, 3, 2)
plt.hist(scaled_height__MinMax, bins=10, alpha=0.5)
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('MinMaxScaler')

# RobustScaler
plt.subplot(1, 3, 3)
plt.hist(scaled_height__Robust, bins=10, alpha=0.5)
plt.xlabel('Scaled Height')
plt.ylabel('Frequency')
plt.title('RobustScaler')

# Add labels and legend
plt.suptitle('Scaling Histograms for Height')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Height")
plt.show()


### Weight
## Fit and transform the weight values using each scaler instance
scaled_weight__Standard = StandardScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])
scaled_weight__MinMax = MinMaxScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])
scaled_weight__Robust = RobustScaler().fit_transform(df[['Weight (Pounds)', 'BMI']])

plt.figure(figsize=(11,5))     # Set figure

## Create histograms for each scaled weight value
# StandardScaler
plt.subplot(1, 3, 1)
plt.hist(scaled_weight__Standard, bins=10, alpha=0.5)
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('StandardScaler')

# MinMaxScaler
plt.subplot(1, 3, 2)
plt.hist(scaled_weight__MinMax, bins=10, alpha=0.5)
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('MinMaxScaler')

# RobustScaler
plt.subplot(1, 3, 3)
plt.hist(scaled_weight__Robust, bins=10, alpha=0.5)
plt.xlabel('Scaled Weight')
plt.ylabel('Frequency')
plt.title('RobustScaler')

# Add labels and legend
plt.suptitle('Scaling Histograms for Weight')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Weight")
plt.show()

