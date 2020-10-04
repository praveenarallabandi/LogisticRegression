#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import torch as torch;


# create a tensor variable, this is a constant parameter, we do not need gradient w.r.t. to it
minimum_w=torch.tensor(np.array([1.0,3.0]),requires_grad=False)

#define some function using pytorch operations (note torch. instead of np.) 
# this function is f(w)=||w-minimum||^2, and so has minimum at minimum_w, i.e. at vector [1.0,3.0]
# it is a convex function so has one minimum, no other local minima
def f(w):
    shiftedW=w-minimum_w
    return torch.sum(torch.mul(shiftedW,shiftedW))

#define starting value of W for gradient descent
#here, W is a 2D vector
initialW=np.random.rand(2)

#create a PyTorch tensor variable for w. 
# we will be optimizing over w, finding its best value using gradient descent (df / dw) so we need gradient enabled
w = torch.tensor(initialW,requires_grad=True)

# this will do gradient descent (fancy, adaptive learning rate version called Adam) for us
optimizer = torch.optim.Adam([w],lr=0.001)

for i in range(10000):
    # clear previous gradient calculations
    optimizer.zero_grad()
    # calculate f based on current value
    z=f(w)
    if (i % 100 == 0 ):
        print("Iter: ",i," w: ",w.data.cpu()," f(w): ",z.item())
    # calculate gradient of f w.r.t. w
    z.backward()
    # use the gradient to change w
    optimizer.step()

print("True minimum: "+str(minimum_w.data.cpu()))
print("Found minimum:"+str(w.data.cpu()))
