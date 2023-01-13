# Math behind ML algorithms - with code

# Linear Regression

Consider the example below:

| Living Area   | bedrooms      | Price(1000$) |
| ------------- |:-------------:| -----:       |
| 2400          | 3             |  400         |
| 1600          | 3             |  330         |
| 2400          | 2             |  540

**Objective**: To predict the price of a house given only the living area and no. of bedrooms.  
- Living area and bedrooms represents the input features - X's.  
- Price is the output. - Y

&nbsp;
Think of the hypothesis as the function that maps input features(X) to the corresponding output(y)
We can express the hypothesis as a sum of linear combinations of the features:

$$h_θ (x)=θ_0+θ_1 x_1+θ_2 x_2$$
```
def hypo(x, theta):
    return np.matmul(x, theta)
    
```

    
**where $θ$ are the parameters(Weights)**
&nbsp;  

we can therefore express $h_θ(x)$ as follows:

$$h_θ(x) = \sum\limits_{i=0}^{d}θ^Tx$$


                    
**Loss Function**:
The loss function measures for each $θ$, how close the $h(x^{(i)})'s$ are to their corresponding $y^{(i)}$

$$J(θ) = {1 \over 2} \sum\limits_{i=1}^{n}(h_θ(x)^{(i)} - y^{(i)})^2 $$

- cost function
$$J(θ) = {1 \over 2n} \sum\limits_{i=1}^{n}(h_θ(x)^{(i)} - y^{(i)})^2 $$
where n is the number of sampled data  

** i.e. cost function is the sum of losses from the loss function **
```
def cost_fn(x,y, theta):
    return (hypo(x, theta) - y).T@(hypo(x, theta) - y)/(2*x.shape[0])
```

## 1.LMS Algorithm
&nbsp;  
**Aim:**

We want to choose $θ$ so as to minimize $J(θ)$ 


### Gradient Descent

It starts with some initial $θ$ and continuously updates $θ$ with the update rule below:

$$θ := θ - \alpha {\partial \over \partialθ_j}J(θ)$$
**$j = 0,1,2,3,.....,d$**

Solving for one training example we have:

$${\partial \over \partialθ_j}J(θ) = {\partial \over \partialθ_j}{1 \over 2}(h_θ(x) - y)^2 $$

**Solving the differential using chain rule yields:**

$$        2 . {1 \over 2}(h_θ(x) - y)^2 . {\partial \over \partialθ_j}(h_θ(x) - y) $$
$$       (h_θ(x) - y) . {\partial \over \partialθ_j}(h_θ(x) - y) $$
$$       (h_θ(x) - y) . {\partial \over \partialθ_j} (\sum\limits_{i=0}^{d} θ_ix_i-y)$$

$$ (h_\theta(x) - y). x_j$$


**LMS(Least mean squares) Update rule/Widrow Hoff learning rule :**

For a single training example:
$$\theta_j := \theta_j - \alpha (y^{(i)} - h_\theta(x^{(i)})).x_j)$$


**Batch Gradient Descent**

For more than one example:

&nbsp;  

Repeat until convergence{$$\theta_j := \theta_j - \alpha \sum\limits_{i=1}^{n}(y^{(i)} - h_\theta(x^{(i)})).x_j^{(i)}),(For Every j)$$}


In a more succint way:

$$\theta := \theta - \alpha \sum\limits_{i=1}^{n}(y^{(i)} - h_\theta(x^{(i)})).x^{(i)}$$

&nbsp;

- Batch Gradient Descent is computationally expensive for a large trainig set since for each step, gradient descent is computed using the entire training set

**Stochastich Gradient Descent**

Loop{  

&nbsp; for $i = 1$ to $n$,&nbsp;{ $$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})).x_j^{(i)}), (For Every j)$$ 
&nbsp;  }  
}

In a more succint way:


$$\theta := \theta + \alpha (y^{(i)} - h_\theta(x^{(i)})).x^{(i)})$$

- This converges much faster
- It is better for a larger training set
- Gradient Descent is computed with a single training sample i.e. k is uniformly sampled at random for entire training set


```
def grad_d(x,y,theta,alpha=0.1 ,epochs=10):
    m = x.shape[0]
    cost = []
    
    for i in range(epochs):
        h_x   = hypo(x , theta)
        error = (1/m)*(x.T@(h_x - y))
        theta = theta - (alpha)*error
        cost.append(cost_fn(x,y,theta))
    
    return theta , cost
     
```


## 2.The Normal Equation

Apart from Gradient Desscent, We can be able to find the closed form solution of $\theta$ that minimizes $J(\theta)$  
- From the training set we can define a design matrix $X$ such that:  

$$
X =
	\begin{bmatrix} 
	x^{(1)T} \\
	x^{(2)T} \\
       .     \\
       .     \\
       .     \\
    x^{(n)T} \\
	\end{bmatrix}
	\quad
$$

- From the target values:

$$ \vec{y} = 
	\begin{bmatrix} 
	y^{(1)} \\
	y^{(2)} \\
    .       \\
    .       \\
	y^{(n)} \\
	\end{bmatrix}
	\quad
	$$
    
  
  
$$ 
\\
X\theta - \vec{y} = 
	\begin{bmatrix} 
	x^{(1)T}\theta - y^{(1)} \\
	x^{(2)T}\theta - y^{(2)} \\
    .       \\
    .       \\
	x^{(n)T}\theta - y^{(n)} \\
	\end{bmatrix}
	\quad
	$$
    
- The cost function is given by:  
$$J(θ) = {1 \over 2} \sum\limits_{i=1}^{n}(h_θ(x)^{(i)} - y^{(i)})^2 $$
$Since:  z^{T}z = z^2$  $and$ $h_θ(x) = X\theta$, we have :
$$J(θ) = {1 \over 2} (X\theta - \vec{y})^{T}(X\theta - \vec{y})$$
 
$$\nabla_\theta J(\theta) = \nabla_\theta {1 \over 2} (X\theta - \vec{y})^{T}(X\theta - \vec{y})$$
$$\nabla_\theta {1 \over 2} (X\theta)^TX\theta - (X\theta)^T\vec{y} - \vec{y}(X\theta)+ \vec{y}^T\vec{y}$$
$Since: \vec{y}^T(X\theta) = (X\theta)^T\vec{y}$, we have :
$$\nabla_\theta {1 \over 2} \theta^T (X^TX)\theta - 2\theta^T(X^T\vec{y}) + \vec{y}^T\vec{y} $$
$Since: \nabla_xx^TAx = 2Ax$, we have :
$$ {1 \over 2}(2X^TX\theta - 2X^T\vec{y}) $$

- To find minima
$$X^TX\theta - X^T\vec{y} = 0$$
$$X^TX\theta = X^T\vec{y}$$  
- Therefore the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equaton below:
$$\theta = (X^TX)^{-1}X^T\vec{y}$$
```

theta_normal = (np.linalg.inv(x.T@x))@(x.T)@(y)

```

## 3.Probabilistic Interpretation

$$y^{(i)} = \theta^TX^{(i)} + \epsilon^{(i)}$$
where $\epsilon$ random noise distributed normally with mean $0$ and variance $\sigma^{2}$


$$ \epsilon^{(i)} = y^{(i)} - \theta^Tx^{(i)} \sim N(0, \sigma^{2})$$ 
$$y^{(i)} \sim N(\theta^Tx^{(i)}, \sigma^{2})$$ 
- This is due to the location scale property of gaussian

$$ P(y^{(i)}| x^{(i)}; \theta ) = \frac{1}{{\sigma \sqrt {2\pi } }}exp \left({-1\over2}{{ (y^{(i)}- \theta^Tx^{(i)}) ^2 } \over {\sigma ^2}}\right) $$


**Likelihood**
$$L(\theta;X,\vec{y}) = p(\vec{y}|X; \theta)$$

$$L(\theta) =  \prod\limits_{i=1}^{n} \frac{1}{{\sigma \sqrt {2\pi } }}exp \left({-1\over2}{{ (y^{(i)}- \theta^Tx^{(i)}) ^2 } \over {\sigma ^2}}\right) $$

**Log Likelihood**

$$ log \;  L(\theta) = {\boldsymbol\ell}(\theta) = \sum\limits_{i=1}^{n} log\left[ \frac{1}{{ \sqrt {2\pi\sigma } }}exp \left({-1\over2}{{ (y^{(i)}- \theta^Tx^{(i)}) ^2 } \over {\sigma ^2}}\right)\right]$$

$$ \sum\limits_{i=1}^{n}  K - {1\over{2\sigma^2}}{ (y^{(i)}- \theta^Tx^{(i)}) ^2 }  $$

$$ {\boldsymbol\ell}(\theta) = nK - {1\over\sigma^2} \left[ \sum\limits_{i=1}^n {1\over2}{ (y^{(i)}- \theta^Tx^{(i)}) ^2 } \right ] $$

**Note**
- Maximimizing the log likelihood minimizes the loss.

$$argmax\; {\boldsymbol\ell}(\theta) = argmin\; J(\theta) $$

## The Code

```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#import data
data = pd.read_csv("USA_Housing.csv")
data.drop('Address',axis= 1, inplace =True)

#split data
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']


df = pd.DataFrame()

#preprocessing

#normalization
for i in range(0,5):
    col = X.columns[i]
    mean = X[col].mean()
    std =X[col].std()
     
    def standard(x):
        return (x-mean)/std
    
    df[col] = X[col].apply(standard)
    
    
x = np.array(df)
y = np.array(y)

x = np.hstack((np.ones((x.shape[0],1)), x))
y = np.reshape(y, (5000,1))

#initial theta values
theta = np.zeros((x.shape[1], 1))

#hypothesis
def hypo(x, theta):
    return np.matmul(x, theta)

#cost function
def cost_fn(x,y, theta):
    return (hypo(x, theta) - y).T@(hypo(x, theta) - y)/(2*x.shape[0])

#gradient descent
def grad_d(x,y,theta,alpha=0.1 ,epochs=10):
    m = x.shape[0]
    cost = []
    
    for i in range(epochs):
        h_x   = hypo(x , theta)
        error = (1/m)*(x.T@(h_x - y))
        theta = theta - (alpha)*error
        cost.append(cost_fn(x,y,theta))
    
    return theta , cost


#function to plot cost
def pl_cost(J, epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(epochs, J)

    
#optimization
theta, cost = grad_d(x, y, theta , alpha= 0.1 , epochs = 30)


epochs = []
z = []
count = 0
for i in cost:
    z.append(i[0][0])
    epochs.append(count)
    count += 1
z = np.array(z)
epochs = np.array(epochs)
pl_cost(z, epochs)


#Testing new sample
sample= data.sample()
sample.drop('Price',axis=1,inplace=True)


for i in range(0,5):
    col = X.columns[i]
    mean = X[col].mean()
    std =X[col].std()
    
    sample_col = sample.columns[i]
     
    def standard(x):
        return (x-mean)/std
    
    sample[sample_col] = sample[sample_col].apply(standard)
        
sample = np.array(sample)
sample[0][0] = 1

result = theta[0] + theta[1]*sample[0][1] + theta[2]*sample[0][2] + theta[3]*sample[0][3]+ theta[4]*sample[0][4]
print("Price of house: ", result[0])


# The normal Equation
theta_normal = (np.linalg.inv(x.T@x))@(x.T)@(y)

result_normal = theta_normal[0] + theta_normal[1]*sample[0][1] + theta_normal[2]*sample[0][2] + theta_normal[3]*sample[0][3]+ theta_normal[4]*sample[0][4]
print("Price of house: ", result_normal[0])


```
### output

![screenshot](https://github.com/Jayem-11/ML-Algorithms-from-scratch/blob/main/Linear_regression/Screenshot%20(549).png)

