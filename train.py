import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import input_data
import time
#from tensorflow.examples.tutorials.mnist import input_data

def show_im(frame):
    frame = frame.reshape(sl,sl)
    imgplot=plt.imshow(frame, interpolation="nearest",cmap="Greys_r")
    plt.show()

def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x)); # Note: there is a typo on this line in the video


mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)
batch_size = 10

np.random.seed(2)
syn0 = 2*np.random.random((785,13)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((14,10)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.



augment=np.ones((batch_size,1))
for j in range(30):
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        l0 = np.hstack((batch_xs,augment))
        for k in xrange(1):
            l1 = nonlin(np.dot(l0, syn0))
            l1 = np.hstack((l1,augment))
            l2 = nonlin(np.dot(l1, syn1))
             # Back propagation of errors using the chain rule.
            l2_error = batch_ys - l2
            #print l2_error
            #print l2
            l2_delta = l2_error*nonlin(l2, deriv=True)#(batch_ys - l2)*(l2*(1-l2))
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * nonlin(l1,deriv=True)
            #update weights (no learning rate term)
            syn1 += l1.T.dot(l2_delta)*0.5
            syn0 += l0.T.dot(l1_delta[:,xrange(13)])*0.5
            #print("Error: " + str(np.mean(np.abs(l2_error))))

    print("Error: " + str(np.mean(np.abs(l2_error))))
print("Output trained syn")
print syn0
print syn1

np.save('syn0', syn0)
np.save('syn1', syn1)







#im=X[0:18].sum(axis=0)
#show_im(im)
