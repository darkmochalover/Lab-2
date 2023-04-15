import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stat

# (0) read excel file
df = pd.read_excel('data/bmi_data_phw1.xlsx')
print(df)

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
model.fit(df[['Height (Inches)']], df[['Weight (Pounds)']])

# Use the fitted model to predict the missing values in the part with dirty values
pred_w = model.predict(df[['Height (Inches)']])     # w'
target_w = df[['Weight (Pounds)']]                  # w


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

z_score = (e_list - mean)/ std

# z_score = np.array(z_score)
# print(e_list)
# print(z_score)

plt.hist(z_score, bins=10)
plt.xlabel('Normalized e values')
plt.ylabel('Frequency')
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

print("[BMI=0] Extremely weak data is: ", extremely_weak)
print("[BMI=4] Obesity data is: ", obesity)

