import re
import numpy as np
import input_data
import time
import cv2
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
syn0 = 2*np.random.random((785,40)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((40,15)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
syn2 = 2*np.random.random((15,10)) - 1

ker=np.ones((3,3))/9

augment=np.ones((batch_size,1))

for j in range(60):
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        l0 = np.hstack((batch_xs,augment))
        for k in xrange(1):
            l1 = nonlin(np.dot(l0, syn0))
            l2 = nonlin(np.dot(l1, syn1))
            l3 = nonlin(np.dot(l2, syn2))

             # Back propagation of errors using the chain rule.
            l3_error = batch_ys - l3

            l3_delta = l3_error*nonlin(l3, deriv=True)#(batch_ys - l2)*(l2*(1-l2))
            l2_error = l3_delta.dot(syn2.T)

            l2_delta = l2_error*nonlin(l2, deriv=True)#(batch_ys - l2)*(l2*(1-l2))
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * nonlin(l1,deriv=True)
            #update weights (no learning rate term)
            syn2 += l2.T.dot(l3_delta)*0.01
            syn1 += l1.T.dot(l2_delta)*0.01
            syn0 += l0.T.dot(l1_delta)*0.01
            #print("Error: " + str(np.mean(np.abs(l2_error))))
    print("Error: " + str(np.mean(np.abs(l3_error))),",iteration: ",j)
print("Output trained syn")
print syn0
print syn1

np.save('syn0', syn0)
np.save('syn1', syn1)
np.save('syn2', syn2)






#im=X[0:18].sum(axis=0)
#show_im(im)
