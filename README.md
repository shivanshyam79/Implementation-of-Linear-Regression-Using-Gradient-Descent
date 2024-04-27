# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
/*
Program to implement the linear regression using gradient descent.
Developed by: shyam R
RegisterNumber: 212223040200
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
```
## Output:
## PROFIT PREDICTION GRAPH:
![268481041-c99c236f-fbd2-4b93-b94d-88b2bae00e67](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/b22f514c-7195-43b3-8b68-9f7945afa988)
## COMPUTE COST VALUE:
![268481067-8ac9bf73-a895-4de9-88ce-b579fe141387](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/de3698d6-b5d2-4421-91bc-6b9919d8e4ad)
## h(x) VALUE:
![268481071-8da5cc0c-9dc2-42a1-baae-175734b8cae7](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/b0184f03-2b63-4e25-b236-8cdb9b0f6fce)
## COST FUNCTION USING GRADIENT DESCENT GRAPH:
![268481088-1b30479a-e925-4dbf-ba31-385d3821b8c0](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/ccc2036b-ad8c-49bc-8164-b81631a1b326)
## PROFIT PREDICTION GRAPH:
![268481090-2a1fe829-35cf-4a3c-89d5-7cbb307dea5c](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/189d422c-43a7-4ea5-97a7-744f5147ff96)
## PROFIT FOR THE POPULATION 35,000:
![268481128-2c26ebc8-59d7-43dc-bd4a-68ef58996b35](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/f80e9f58-4015-43b9-af7d-babdb1ed558a)
## PROFIT FOR THE POPULATION 70,000:
![268481136-30fd7a04-a6d2-4a03-a17f-53340d4f6363](https://github.com/shivanshyam79/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/151513860/367c22b6-5f2d-4d76-9b99-132e90e94011)








## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
