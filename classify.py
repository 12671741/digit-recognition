import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import input_data
import cv2
sl=28
syn0 = np.load('syn0.npy')
syn1 = np.load('syn1.npy')
def show_im(frame):
    frame = frame[0:784].reshape(sl,sl)
    #frame = gaussian_filter(frame, sigma=2)
    imgplot=cv2.imshow("name",frame)
    cv2.waitKey(0)

def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video

def classify(frame):
    l1 = nonlin(np.dot(frame, syn0))
    l1=np.append(l1,1)
    l2 = nonlin(np.dot(l1, syn1))
    return l2

mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)
batch_size = 40
total_batch = int(mnist.test.num_examples/batch_size)
augment=np.ones((batch_size,1))
ticks=0
k=0
for i in range(total_batch):
    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
    batch_xs = np.hstack((batch_xs,augment))
    for j in xrange(batch_size):
        c=classify(batch_xs[j])
        #print np.argmax(c),c.max()
        k=k+1
        #print c
        #show_im(batch_xs[j])
        if np.argmax(c)==np.argmax(batch_ys[j]):
            ticks=ticks+1
        #    print "correct"
        #else:
        #    print "wrong"
print "correct ",ticks
print float(ticks)/mnist.test.num_examples
print k
