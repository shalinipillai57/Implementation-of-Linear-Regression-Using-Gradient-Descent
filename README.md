# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters
2.Compute predicted values
3.Update parameters using Gradient Descent
4. Predict profit

## Program:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Startup.csv")

X = data['R&D Spend'].values
y = data['Profit'].values


X = (X - X.mean()) / X.std()


m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)


for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)


y_pred = m * X + b


plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()

## Output:
<img width="645" height="491" alt="Screenshot 2026-04-22 090833" src="https://github.com/user-attachments/assets/106d86ee-7e0f-4c91-b96d-ef97b3de02ea" /> 


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
