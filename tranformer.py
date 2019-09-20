
#%%
""" For feature engineering """
import csv
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
""" For model fitting """
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os

print(os.getcwd())

attributes = ("WindSpeed","WindDirection","Temperature","RelativeHumindity","BarometricPressure")

headers = [str(y) + str(x) for x in attributes for y in ["sl_","m_","std_"]]
headers.append("PM2.5(ug/m3)")

df_data = pd.read_csv("train_data.csv", sep =',' , header = 0)

with open('syn_data.csv', mode='w', newline='') as conv_file:
    writer = csv.writer(conv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    
    timespan = {"t_WindSpeed" : 12 , "t_WindDirection" : 12 , "t_Temperature" : 12,
                "t_RelativeHumindity" : 12 , "t_BarometricPressure" : 12 , "t_RainVolume" : 12 }

    
    count_row,count_col = df_data.shape
    maxSpan = max(timespan.values())
    
    for i in range(count_row):
        X = df_data.values[i:i+maxSpan:,1:7].astype("float64")
        Y = df_data.values[i:i+maxSpan,7].astype("float64")
        shift = df_data.values[i+12+11 , 7]

        if (i+maxSpan+12) == count_row:
                break

        buffer = []

        x_ws = X[maxSpan - timespan["t_WindSpeed"] : len(X) ,0]
        y_ws = Y[maxSpan - timespan["t_WindSpeed"] : len(X)]
        slope,_,_,_,_ = linregress(x_ws, y_ws)
        buffer.append(str(round(slope,4)))
        buffer.append(str(round(np.mean(x_ws),4)))
        buffer.append(str(round(np.std(x_ws),4)))

        x_wd = X[maxSpan - timespan["t_WindDirection"] : len(X) ,1]
        y_wd = Y[maxSpan - timespan["t_WindDirection"] : len(X)]
        slope,_,_,_,_ = linregress(x_wd, y_wd)
        buffer.append(str(round(slope,4)))
        buffer.append(str(round(np.mean(x_wd),4)))
        buffer.append(str(round(np.std(x_wd),4)))

        x_temp = X[maxSpan - timespan["t_Temperature"] : len(X) ,2]
        y_temp = Y[maxSpan - timespan["t_Temperature"] : len(X)]
        slope,_,_,_,_ = linregress(x_temp, y_temp)
        buffer.append(str(round(slope,4)))
        buffer.append(str(round(np.mean(x_temp),4)))
        buffer.append(str(round(np.std(x_temp),4)))
                
        x_rh = X[maxSpan - timespan["t_RelativeHumindity"] : len(X) ,3]
        y_rh = Y[maxSpan - timespan["t_RelativeHumindity"] : len(X)]
        slope,_,_,_,_ = linregress(x_rh, y_rh)
        buffer.append(str(round(slope,4)))
        buffer.append(str(round(np.mean(x_rh),4)))
        buffer.append(str(round(np.std(x_rh),4)))
        
        x_bp = X[maxSpan - timespan["t_BarometricPressure"] : len(X) ,4]
        y_bp = Y[maxSpan - timespan["t_BarometricPressure"] : len(X)]
        slope,_,_,_,_ = linregress(x_bp, y_bp)
        buffer.append(str(round(slope,4)))
        buffer.append(str(round(np.mean(x_bp),4)))
        buffer.append(str(round(np.std(x_bp),4)))
        
        # x_rv = X[maxSpan - timespan["t_RainVolume"] : len(X) ,5]
        # y_rv = Y[maxSpan - timespan["t_RainVolume"] : len(X)]
        # slope,_,_,_,_ = linregress(x_rv, y_rv)
        # buffer.append(str(round(slope,4)))
        # buffer.append(str(round(np.mean(x_rv),4)))
        # buffer.append(str(round(np.std(x_rv),4)))
        
        buffer.append(shift)
        writer.writerow(buffer)
        


""" 
Model Fitting Section
"""
df_data = pd.read_csv("syn_data.csv", sep =',' , header = 0)
X = df_data.values[:,0:15]
Y = df_data.values[:,15]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators = 91,criterion="mse", random_state = 6)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred = np.around(y_pred)

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df1.head(25))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#%%
def evaluateAccuracy(numlist):
        timespan.update(zip(timespan,numlist))

