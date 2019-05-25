import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#import dataframe from csv input
ip = pd.read_csv('Canada_PCI.csv')

#create linear regression object
reg = linear_model.LinearRegression()
reg.fit(ip[['year']], ip.per_capita_income)

#input new csv file to predict
df = pd.read_csv('years.csv')
p = reg.predict(df)

#create a new column in csv file
df['PCI'] = p

#save the ouput to new csv file
df.to_csv('prediction.csv', index=False)

#plotting a linear model to the given input
plt.xlabel('year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(ip.year, ip.per_capita_income, color='red')
plt.plot(ip.year, reg.predict(ip[['year']]), color='blue')
plt.show(block=True)

