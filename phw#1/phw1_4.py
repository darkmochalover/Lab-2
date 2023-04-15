import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data file from excel file
df = pd.read_excel("data/bmi_data_phw1.xlsx")

# (equation E) linear regression 
x = df['Height (Inches)']
y = df['Weight (Pounds)']
slope, intercept = np.polyfit(x, y, 1)

# For (height h, weight w) of each record, compute e=w-w’, where w’ is obtained for h using E
df['w_prime'] = df['Height (Inches)'] * slope + intercept
df['e'] = df['Weight (Pounds)'] - df['w_prime']

# Normalize the e values, i.e., compute ze=[e-μ(e)]/σ(e)
e_mean = df['e'].mean()
e_std = df['e'].std()
df['ze'] = (df['e'] - e_mean) / e_std

# Decide a value α (≥0); for records with ze<-α, set BMI = 0; for those with ze>α, set BMI = 4
alpha = 2.0
df.loc[df['ze'] < -alpha, 'BMI'] = 0
df.loc[df['ze'] > alpha, 'BMI'] = 4

# actual BMI and predicted BMI 계산
df['Actual BMI'] = df['Weight (Pounds)'] / (df['Height (Inches)'] ** 2) * 703
df['Predicted BMI'] = df['w_prime'] / (df['Height (Inches)'] ** 2) * 703

# Show histogram
plt.hist(df['ze'], bins=10)
plt.xlabel('Normalized error (ze)')
plt.ylabel('Frequency')
plt.title('Distribution of ze')
plt.show()

# Compare actual BMI and predicted BMI
plt.hist(df['Actual BMI'], bins=10, alpha=0.5, label='Actual BMI')
plt.hist(df['Predicted BMI'], bins=10, alpha=0.5, label='Predicted BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of Actual BMI and Predicted BMI')
plt.legend()
plt.show()
