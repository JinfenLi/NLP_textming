import torch
from torch.autograd import Variable
import numpy as np
a = torch.Tensor([[1,2],[3,4]])
# Variables allow you to wrap a Tensor and record operations performed on it
b = Variable(torch.Tensor([[1,2],[3,4]]),requires_grad=True)
print(b)
y = torch.sum(b**2)
y.backward()# compute gradients of y wrt b
print(b.grad)

import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self,x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 32, 100, 50, 10
x = Variable(torch.randn(N, D_in))
model = TwoLayerNet(D_in, H, D_out)
y_pred = model(x) # 32x10

def myCrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]
    outputs = F.log_softmax(outputs, dim =1) # 1: row 0: column
    outputs = outputs(range(batch_size), labels) # pick the values corresponding to the labels
    return -torch.sum(outputs)/num_examples

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

def accuracy(out, labels):
    outputs = np.argmax(out, axis = 1)
    return np.sum(outputs == labels)/float(labels.size)

'''
import tensorflow as tf
m1 = tf.constant([3,5])
m2 = tf.constant([2,4])
result = tf.add(m1,m2)
sess = tf.Session()
with sess.as_default():
    print(result.eval())
with tf.Session() as sess:
    print(sess.run(result))    
'''

