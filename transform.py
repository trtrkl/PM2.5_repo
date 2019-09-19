import csv
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

df_data = pd.read_csv("train_data.csv", sep =',' , header = 0)
count_row,count_col = df_data.shape

timespan = {"t_WindSpeed" : 12 , "t_WindDirection" : 12 , "t_Temperature" : 12,
            "t_RelativeHumindity" : 12 , "t_BarometricPressure" : 12 , "t_Rain Volume" : 12 }

attributes = ("WindSpeed","WindDirection","Temperature","RelativeHumindity","BarometricPressure","Rain Volume")



with open('syn_data.csv', mode='w', newline='') as conv_file:
    writer = csv.writer(conv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(x) + str(y) for x,y in for x in (["sl_","m_","std_"] for y in attributes ].append("PM2.5(ug/m3)"))      #Head of the data table
   
    for i in range(count_row):

        span = max(timespan.values())

        print("index : " + str(i))
        print("max span : " + span)
        print("index + max span + 11 : " + str(i+span+11))

        dict.update

        if (i+span)+12 == count_row:
            break


        time = df_data.values[i:i+span , 0]
        X = df_data.values[i:i+span , 1:7].astype("float64")
        Y = df_data.values[i:i+span , 7].astype("float64")
        shift = df_data.values[i+span+11 , 7]

        print("Mean : " + str(np.mean(X)))
        print("STD : " + str(np.std(X)))

        slope,intercept,r,p,err = linregress(X, Y) 

        print("Slope : "+ str(slope) + "\n")

        mintime = str(min(time)).split(",")
        maxtime = str(max(time)).split(",")
        minstr = mintime[2] + "-"+ mintime[1] + " @ " + mintime[3]
        maxstr = maxtime[2] + "-"+ maxtime[1] + " @ " + maxtime[3]
        
        buffer = [minstr + " : " + maxstr]      #Add time range
        buffer.append(str(round(slope,4)))      #Add slope
        buffer.append(str(round(np.mean(X),4))) #Add mean
        buffer.append(str(round(np.std(X),4)))  #Add standard deviation
        buffer.append(str(shift))               #Add prediction time
        
        writer.writerow(buffer)

      
        print("Bufer : " + str(buffer) + "\n")


""" 
    plt.plot(X, Y, 'o', label='original data')
    plt.plot(X, intercept + slope*X, 'r', label='fitted line')
    plt.xlabel('Barometric Pressure', fontsize=18)
    plt.ylabel('PM2.5', fontsize=16)
    plt.legend()
    plt.show() """
print("\n")
print(count_row)