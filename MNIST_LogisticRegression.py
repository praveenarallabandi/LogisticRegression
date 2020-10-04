import torch as torch
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

# the 28×28 sized images will be 784 pixel input values
totalPixelsInputValues=784

def load_data():
    (x_train, y_train), (x_test, y_test)= mnist.load_data()
    return x_train, y_train, x_test, y_test

# load dataset from keras
(x_train, y_train), (x_test, y_test)= mnist.load_data()
# x_train, y_train, x_test, y_test = load_data()
X_Train = torch.from_numpy(x_train)
Y_Train = torch.from_numpy(y_train)
X_Test = torch.from_numpy(x_test)
Y_Test = torch.from_numpy(y_test)


# Filtered dataset with class values 3 and 5 for Vid: V00933455
train_filter_class= torch.where((Y_Train==5) | (Y_Train==3))
test_filter_class= torch.where((Y_Test==5) | (Y_Test==3))
x_train, y_train = x_train[train_filter_class], y_train[train_filter_class]
x_test, y_test = x_test[test_filter_class], y_test[test_filter_class]
""" print('x_train', x_train)
print('y_train', y_train) """
# reshaping train and test data
x_train = x_train.reshape(x_train.shape[0],784) / 255
x_test = x_test.reshape(x_test.shape[0],784) / 255

#changing classes to 0's and 1's as in binary classification

for j in range(y_train.shape[0]):
    if y_train[j]==3:
        y_train[j]=1

for j in range(y_test.shape[0]):
    if y_test[j]==3:
        y_test[j]=1

#selecting samples
X=x_train
Y=y_train
# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:

    mu_prior_cov = 100*np.eye(totalPixelsInputValues)
    mu_prior_mu = np.zeros((totalPixelsInputValues,))

    # Priors for w,b (Gaussian priors), centered at 0, with very large std.dev.
    w = pm.MvNormal('estimated_w', mu=mu_prior_mu, cov=mu_prior_cov, shape=totalPixelsInputValues)
    b  = pm.Normal('estimated_b',0,100)
    # calculate u=w^Tx+b
    u = pm.Deterministic('my_u',T.dot(X,w) + b)
    prob = pm.Deterministic('my_prob',1.0 / (1.0 + T.exp(-1.0*u)))
    Y_obs=pm.Bernoulli('Y_obs',p=prob,observed = Y)

# done with setting up the model

map_estimate1 = pm.find_MAP(model=basic_model)

true_w= map_estimate1.get("estimated_w")
true_b= map_estimate1.get("estimated_b")
w_trans=np.transpose(true_w)

# Predicting class for test data
def predict_classes(x):
    true_u = np.add(np.dot(x,w_trans),true_b)
    prob_test= 1.0/(1.0+np.exp(-1.0*true_u))
    if prob_test>=0.5:
        probPred=1
    else:
        probPred=0
    return probPred

X_test=x_test
Y_test=y_test
predictions = np.zeros(Y_test.shape[0])  #initializing predictions array

for i in range(X_test.shape[0]):
    collectProbPred=predict_classes(X_test[i])
    predictions[i] = collectProbPred

accuracy_test = accuracy_score(Y_test, predictions)
print("Accuracy: ",accuracy_test * 100 ,"%")
