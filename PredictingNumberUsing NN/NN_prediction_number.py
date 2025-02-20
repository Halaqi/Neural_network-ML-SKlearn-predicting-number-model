import numpy as np
import matplotlib.pyplot as plt
import random

train_x = np.loadtxt(r"D:\programming\Machine learning\PredictingNumberUsing NN\train_X.csv", delimiter=',').T
train_y = np.loadtxt(r"D:\programming\Machine learning\PredictingNumberUsing NN\train_label.csv", delimiter=',').T
test_x = np.loadtxt(r"D:\programming\Machine learning\PredictingNumberUsing NN\test_X.csv", delimiter=',').T
test_y = np.loadtxt(r"D:\programming\Machine learning\PredictingNumberUsing NN\test_label.csv", delimiter= ',').T


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)



def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    expX = np.exp(x)
    return expX/np.sum(expX, axis = 0)

def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))

def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)

# initializing parameters 
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    
    w2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }
    
    return parameters

# forward propagation
def forward_propagation(x, parameters):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    z1 = np.dot(w1, x) + b1
    a1 = tanh(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    
    forward_cache = {
        "z1" : z1,
        "a1" : a1,
        "z2" : z2,
        "a2" : a2
    }
    
    return forward_cache



# computing the loss cost
def cost_function(a2, y):
    m = y.shape[1]
    
    cost = -(1/m)*np.sum(y*np.log(a2))
    
    return cost

# backward propagation
def backward_propagation(x, y, parameters, forward_cache):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    a1 = forward_cache['a1']
    a2 = forward_cache['a2']
    
    m = x.shape[1]
    
    dz2 = (a2 - y)
    dw2 = (1/m)*np.dot(dz2, a1.T)
    db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
    
    dz1 = (1/m)*np.dot(w2.T, dz2)* derivative_relu(a1)
    dw1 = (1/m)*np.dot(dz1, x.T)
    db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)
    
    gradients = {
        "dw1" : dw1,
        "db1" : db1,
        "dw2" : dw2,
        "db2" : db2
    }
    
    return gradients

# updating the parameters
def update_parameters(parameters, gradients, learning_rate):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']
    
    w1 = w1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2
    
    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }
    
    return parameters
# NN Model
def model(x, y, n_h, learning_rate, iterations):
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    
    cost_list = []
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(iterations):
        
        forward_cache = forward_propagation(x, parameters)
        
        cost = cost_function(forward_cache['a2'], y)
        
        gradients = backward_propagation(x, y, parameters, forward_cache)
        
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("Cost after", i, "iterations is :", cost)
        
    return parameters, cost_list
    
# calling the model function
iterations = 500
n_h = 1000
learning_rate = 0.02

Parameters, Cost_list = model(train_x, train_y, n_h = n_h, learning_rate = learning_rate, iterations = iterations)

plt.plot(np.arange(iterations), Cost_list)
plt.show()
    
# testing the data and checking the accuracy
def accuracy(inputs, outputs, Parameters):
    forward = forward_propagation(inputs, Parameters)
    a_out = forward['a2']
    a_out = np.argmax(a_out, 0)
    
    outputs = np.argmax(outputs, 0)
    
    acc = np.mean(a_out == outputs) * 100
    
    return acc

# calling the  accuracy function
print('the accuracy of our trainning data is', accuracy(train_x, train_y, Parameters) ,"%")
print('the accuracy of our testing data is', accuracy(test_x, test_y, Parameters) ,"%")
    
# displaying image and predicting the output
idx = int(random.randrange(0,test_x.shape[1]))
plt.imshow(test_x[:, idx].reshape((28,28)),cmap='gray')
plt.show()

cache = forward_propagation(test_x[:, idx].reshape(test_x[:, idx].shape[0], 1), Parameters)
a_pred = cache['a2']  
a_pred = np.argmax(a_pred, 0)

print("Our model says it is :", a_pred[0])  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    