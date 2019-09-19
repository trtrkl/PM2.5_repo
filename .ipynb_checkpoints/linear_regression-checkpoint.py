import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import re

df_data = pd.read_csv("syn_data.csv", sep =',' , header = 0)
X = df_data.values[:,1:4]
Y = df_data.values[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred = np.around(y_pred)

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df1.head(25))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficients: \n', regressor.coef_)

""" # Scatter Plot
for i,j in {"1":"Wind Speed(m/s)" , "2":"Wind Derection(Degree)" ,  "3":"Temperature(Celsius)"  , "4":"Relative Humidity(%)", "5":"Barometric Pressure(mmHG)" , "6":"Rain Volume(mm)" }.items():
    x = df_data.values[:,int(i)]
    y = df_data.values[:,7]
    
    m = re.search('(\w|\s)+',j).group(0)

    colors = (0,0,0)
    area = np.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot : {} to {}'.format(j,"PM2.5(ug/m3)"))
    plt.xlabel(j)
    plt.ylabel('PM2.5')
    plt.savefig("sc-{} to {}.jpg".format(m,"PM2.5"))
    plt.clf()
 """