import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression

# (0) read excel file
df = pd.read_excel('data/bmi_data_phw1.xlsx')


'''
PHW 1-4
* Divide the input dataset D into two groups D_f and D_m according to gender
* Do the same as done previously for each of D_f and D_m
* Compare your BMI estimates with the actual BMI values in the given dataset
'''

D_f = df[df['Sex'] == 'Female']
D_m = df[df['Sex'] == 'Male']

print(D_f)
print(D_m)


# For Female Data
'''
PHW1-1
Data Exploration
* Print dataset statistical data, feature names & data types
* Plot height & weight histograms (bins=10) for each BMI value
* Plot scaling results for height and weight
'''

# Print statistical data
print(D_f.describe())

# print feature names
print(D_f.columns)

# print data type
print(D_f.dtypes)

# plot weight histogram for each BMI value
fig = plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(D_f[D_f.BMI == n]['Weight (Pounds)'],bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Female Weight Histogram")
plt.show()

# plot height histogram for each BMI value
plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(D_f[D_f.BMI == n]['Height (Inches)'],
                bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Female Height Histogram")
plt.show()

# Height
# Fit and transform the height values using each scaler instance
scaled_height__Standard = StandardScaler().fit_transform(
    D_f[['Height (Inches)', 'BMI']])
scaled_height__MinMax = MinMaxScaler().fit_transform(
    D_f[['Height (Inches)', 'BMI']])
scaled_height__Robust = RobustScaler().fit_transform(
    D_f[['Height (Inches)', 'BMI']])

plt.figure(figsize=(11, 5))     # Set figure

# Create histograms for each scaled height value
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
plt.suptitle('Scaling Histograms for Height - Female Data')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Height - Female Data")
plt.show()

# Weight
# Fit and transform the weight values using each scaler instance
scaled_weight__Standard = StandardScaler().fit_transform(
    D_f[['Weight (Pounds)', 'BMI']])
scaled_weight__MinMax = MinMaxScaler().fit_transform(
    D_f[['Weight (Pounds)', 'BMI']])
scaled_weight__Robust = RobustScaler().fit_transform(
    D_f[['Weight (Pounds)', 'BMI']])

plt.figure(figsize=(11, 5))     # Set figure

# Create histograms for each scaled weight value
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
plt.suptitle('Scaling Histograms for Weight - Female Data"')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Weight - Female Data")
plt.show()

'''
PHW1-2
Program: find outlier people
    * Read the Excel dataset file, and compute the linear regression equation E for the input dataset D

    * For (height h, weight w) of each record, compute e = w - w'
        (w' : obtained for h using E)

    * Normalize the e values; and plot a histogram showing the distribution of z (bins=10)
    
    * Decide a value a(>=0); 
    for records with z<-a, set BMI = 0;
    for those with z>a, set BMI = 4
'''

'''
(1) 
* Compute the linear regression equation E for the input dataset D
'''
# Fit a linear regression model to the part without dirty values

model = LinearRegression()
# Input feature: 'Height (Inches)',
# Output feature: 'Weight (Pounds)'
model.fit(D_f[['Height (Inches)']], D_f[['Weight (Pounds)']])

# Use the fitted model to predict the missing values in the part with dirty values
pred_w = model.predict(D_f[['Height (Inches)']])     # w'
target_w = D_f[['Weight (Pounds)']]                  # w

'''
(2) 
* Normalize the e values; and plot a histogram showing the distribution of z (bins=10
'''
# Compute Error
e_list = target_w - pred_w                               # e = w - w'
print("\n\n[Error of Linear Regression Equation]\n", e_list)

e_list = np.array(e_list['Weight (Pounds)'])
mean = np.mean(e_list)
std = np.std(e_list)

z_score = (e_list - mean) / std

# z_score = np.array(z_score)
# print(e_list)
# print(z_score)

plt.hist(z_score, bins=10)
plt.xlabel('Normalized e values - Female')
plt.ylabel('Frequency')
plt.savefig("Normalized e values - Female")
plt.show()

'''
(3)
* Decide a value a(>=0); 
    for records with z<-a, set BMI = 0;
    for those with z>a, set BMI = 4
'''
threshold = 2
extremely_weak = []
obesity = []

for e, z in zip(e_list, z_score):
    if z > threshold:
        obesity.append(e)
    elif z < -threshold:
        extremely_weak.append(e)

print("\n\n[In Female Data . . .]")
print("[BMI=0] Extremely weak data is: ", extremely_weak)
print("[BMI=4] Obesity data is: ", obesity)


# For Male Data
'''
PHW1-1
Data Exploration
* Print dataset statistical data, feature names & data types
* Plot height & weight histograms (bins=10) for each BMI value
* Plot scaling results for height and weight
'''

# Print statistical data
print(D_m.describe())

# print feature names
print(D_m.columns)

# print data type
print(D_m.dtypes)

# plot weight histogram for each BMI value
fig = plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(D_m[D_m.BMI == n]['Weight (Pounds)'],
                bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Male Weight Histogram")
plt.savefig("Male Weight Histogram")
plt.show()

# plot height histogram for each BMI value
plt.figure(figsize=(10, 8))

for n in range(5):
    plt.subplot(2, 3, n+1)
    plt.hist(D_m[D_m.BMI == n]['Height (Inches)'],
                bins=10, label="{}".format(n))
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

plt.suptitle("Male Height Histogram")
plt.savefig("Male Height Histogram")
plt.show()

# Height
# Fit and transform the height values using each scaler instance
scaled_height__Standard = StandardScaler().fit_transform(
    D_m[['Height (Inches)', 'BMI']])
scaled_height__MinMax = MinMaxScaler().fit_transform(
    D_m[['Height (Inches)', 'BMI']])
scaled_height__Robust = RobustScaler().fit_transform(
    D_m[['Height (Inches)', 'BMI']])

plt.figure(figsize=(11, 5))     # Set figure

# Create histograms for each scaled height value
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
plt.suptitle('Scaling Histograms for Height - Male Data')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Height - Male Data")
plt.show()

# Weight
# Fit and transform the weight values using each scaler instance
scaled_weight__Standard = StandardScaler().fit_transform(
    D_m[['Weight (Pounds)', 'BMI']])
scaled_weight__MinMax = MinMaxScaler().fit_transform(
    D_m[['Weight (Pounds)', 'BMI']])
scaled_weight__Robust = RobustScaler().fit_transform(
    D_m[['Weight (Pounds)', 'BMI']])

plt.figure(figsize=(11, 5))     # Set figure

# Create histograms for each scaled weight value
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
plt.suptitle('Scaling Histograms for Weight - Male Data"')
plt.legend(loc='upper right')

# Show the plot
plt.savefig("Scaling Histograms for Weight - Male Data")
plt.show()

'''
PHW1-2
Program: find outlier people
    * Read the Excel dataset file, and compute the linear regression equation E for the input dataset D

    * For (height h, weight w) of each record, compute e = w - w'
        (w' : obtained for h using E)

    * Normalize the e values; and plot a histogram showing the distribution of z (bins=10)
    
    * Decide a value a(>=0); 
    for records with z<-a, set BMI = 0;
    for those with z>a, set BMI = 4
'''

'''
(1) 
* Compute the linear regression equation E for the input dataset D
'''
# Fit a linear regression model to the part without dirty values

model = LinearRegression()
# Input feature: 'Height (Inches)',
# Output feature: 'Weight (Pounds)'
model.fit(D_m[['Height (Inches)']], D_m[['Weight (Pounds)']])

# Use the fitted model to predict the missing values in the part with dirty values
pred_w = model.predict(D_m[['Height (Inches)']])     # w'
target_w = D_m[['Weight (Pounds)']]                  # w

'''
(2) 
* Normalize the e values; and plot a histogram showing the distribution of z (bins=10
'''
# Compute Error
e_list = target_w - pred_w                               # e = w - w'
print("\n\n[Error of Linear Regression Equation]\n", e_list)

e_list = np.array(e_list['Weight (Pounds)'])
mean = np.mean(e_list)
std = np.std(e_list)

z_score = (e_list - mean) / std

# z_score = np.array(z_score)
# print(e_list)
# print(z_score)

plt.hist(z_score, bins=10)
plt.xlabel('Normalized e values - Male')
plt.ylabel('Frequency')
plt.savefig("Normalized e values - Male")
plt.show()

'''
(3)
* Decide a value a(>=0); 
    for records with z<-a, set BMI = 0;
    for those with z>a, set BMI = 4
'''
threshold = 2
extremely_weak = []
obesity = []

for e, z in zip(e_list, z_score):
    if z > threshold:
        obesity.append(e)
    elif z < -threshold:
        extremely_weak.append(e)

print("\n\n[In Male Data . . .]")
print("[BMI=0] Extremely weak data is: ", extremely_weak)
print("[BMI=4] Obesity data is: ", obesity)

# actual BMI and predicted BMI 계산
Pred_BMI = target_w / (df['Height (Inches)'] ** 2) * 703
Actual_BMI = target_w / (df['Height (Inches)'] ** 2) * 703


# actual BMI and predicted BMI 작성
plt.hist(Pred_BMI, bins=10, alpha=0.5, label='Actual BMI')
plt.hist(Actual_BMI, bins=10, alpha=0.5, label='Predicted BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of Actual BMI and Predicted BMI')
plt.legend()
plt.show()