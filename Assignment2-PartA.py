# Design out model (input,output size,forward pass)
# Construct loss and optimizer
# Training loop
#   - forward pass - compute prediction
#   - backward pass - gradients
#   - Update weights

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
X=torch.tensor([-1.67245526,-2.36540279,-2.14724263,1.40539096,1.24297767,-1.71043904,2.31579097,2.40479939,-2.22112823], dtype=torch.float32)
Y=torch.tensor([-18.56122168, -24.99658931, -24.41907817,  -2.688209,
                -1.54725306,  -19.18190097,   1.74117419,
                3.97703338, -24.80977847], dtype=torch.float32)
w1 = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
w2 = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
w3 = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
w4 = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
w5 = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


""" print('w0 = {} w1 = {} w2 = {} w3 = {} w4 = {}'.format(w0, w1, w2, w3, w4)) """

# model prediction
def forward(x):
    result1 = w1 + 0 * x
    result2 = w1 + w2 * x
    result3 = w1 + w2 * x + w3 * (x ** 2)
    result4 = w1 + w2 * x + w3 * (x ** 2) + w4 * (x ** 3)
    result5 = w1 + w2 * x + w3 * (x ** 2) + w4 * (x ** 3) + w5 * (x ** 4)
    return result1, result2, result3, result4, result5
    
# loss 
loss=nn.MSELoss()

#gradient
learning_rate = 0.001
optimizer1=torch.optim.SGD([w1],lr=learning_rate)
optimizer2=torch.optim.SGD([w2],lr=learning_rate)
optimizer3=torch.optim.SGD([w3],lr=learning_rate)
optimizer4=torch.optim.SGD([w4],lr=learning_rate)
optimizer5=torch.optim.SGD([w5],lr=learning_rate)

#Training

n=1000
for epoch in range(n):
    # model prediction
    result1, result2, result3, result4, result5 = forward(X)

    #loss and gradients
    l1=loss(Y,result1)
    l1.backward()
    g = w1.grad.item()
    optimizer1.step()
    optimizer1.zero_grad()

    l2=loss(Y,result2)
    l2.backward()
    optimizer2.step()
    optimizer2.zero_grad()

    l3=loss(Y,result3)
    l3.backward()
    optimizer3.step()
    optimizer3.zero_grad()

    l4=loss(Y,result4)
    l4.backward()
    optimizer4.step()
    optimizer4.zero_grad()

    l5=loss(Y,result5)
    l5.backward()
    optimizer5.step()
    optimizer5.zero_grad()
    
    #update weights

   
    if epoch %100 == 0:
        print(f'epoch {epoch+1}: w1 = {w1:.3f}, w2 = {w2:.3f}, w3 = {w3:.3f}, w4 = {w4:.3f}, w5 = {w5:.3f}')  

#print(predict1, predict2, predict3, predict4, predict5)
print(' result1 = {}\n result2 = {}\n result3 = {}\n result4 = {}\n result5 = {}'.format(result1, result2, result3, result4, result5))

#Plotting
plt.xlabel('X values')
plt.ylabel('Y values/predicted values')
f1 = plt.figure(1)
plt.plot(X,Y,'ro')
plt.plot(X,result1.data,'bo')
plt.title('For n=1')

f2 = plt.figure(2)
plt.plot(X,Y,'ro')
plt.plot(X,result2.data,'bo')
plt.title('For n=2')

f3 = plt.figure(3)
plt.plot(X,Y,'ro')
plt.plot(X,result3.data,'bo')
plt.title('For n=3')

f4 = plt.figure(4)
plt.plot(X,Y,'ro')
plt.plot(X,result4.data,'bo')
plt.title('For n=4')

f5 = plt.figure(5)
plt.plot(X,Y,'ro')
plt.plot(X,result5.data,'bo')
plt.title('For n=5')

plt.show()
