from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics

df_data = pd.read_csv("syn_data.csv", sep =',' , header = 0)

X = df_data.values[:,1:4]
Y = df_data.values[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred = np.around(y_pred.astype(np.double))

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df1.head(25))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


