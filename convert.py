import csv
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

df_data = pd.read_csv("train_data.csv", sep =',' , header = 0)
count_row,count_col = df_data.shape


with open('syn_data.csv', mode='w', newline='') as conv_file:
    writer = csv.writer(conv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Time Range" , "Slope" ,"Mean","STD", "PM2.5(ug/m3)"])      #Head of the data table
   
    for i in range(count_row):

        print("index : " + str(i))
        print("index + 23 : " + str(i+23))

        if i+23 == count_row:
            break

        time = df_data.values[i:i+12 , 0]
        X = df_data.values[i:i+12 , 5].astype("float64")
        Y = df_data.values[i:i+12 , 7].astype("float64")
        shift = df_data.values[i+12+11 , 7]

        print("Mean : " + str(np.mean(X)))
        print("STD : " + str(np.std(X)))

        slope,intercept,r,p,err = linregress(X, Y) 

        print("Slope : "+str(slope) + "\n")

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


print("\n")
print(count_row)