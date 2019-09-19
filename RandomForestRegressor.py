import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

df_data = pd.read_csv("syn_data.csv", sep =',' , header = 0)

X = df_data.values[:,1:4]
Y = df_data.values[:,4]
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

# import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\\bin'
# from sklearn.tree import export_graphviz
# tree = rf.estimators_[28]
# export_graphviz(tree, out_file='tree.dot', feature_names = df_data.columns[1:4].tolist(), rounded = True, proportion = False, precision = 2, filled = True)
# from subprocess import check_call
# check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=2400']) 