import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
dataframe = pd.read_csv('challenge_dataset.txt', header=None)
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# #visualize results
# plt.scatter(x_values, y_values)
# plt.plot(x_values, body_reg.predict(x_values))
# plt.show()

#select a random point
index = np.random.randint(0,x_values.size)
myvalue = x_values[0].T[index]
myactual = y_values[1].T[index] 
print('Random X: {}'.format(myvalue))

# determine prediction from the linear regression
myprediction = body_reg.predict(myvalue).T[0][0]

print('Actual Y: {}'.format(myactual))
print('Predicted Y: {}'.format(myprediction))

print('My error: {}'.format(myprediction - myactual))
print('Model Error: {}'.format(body_reg.coef_))
print('Model Score: {}'.format(body_reg.score(body_reg.predict(x_values), y_values)))
