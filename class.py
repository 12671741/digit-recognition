import tensorflow as tf
import re
import numpy as np
import cv2


def add_data(filename,lenth):
    d=[]
    with open(filename) as f:
        lines = f.readlines()
        lines = ''.join(lines)
        d=re.findall('\-?\d+',lines)
        d=map(np.float32,d)
        d=np.array(d)
        data=d.reshape(len(d)/lenth,lenth)
        data=127*(data+1)
    return data

framelen=1024

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
W = tf.Variable(tf.zeros([784, 10]),name="W")
b = tf.Variable(tf.zeros([10]),name="b")
np.set_printoptions(threshold='nan')

data=add_data('../data/2/data.m',framelen)


saver = tf.train.Saver()
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    # Restore variables from disk.
    saver.restore(sess, "./model/model.ckpt")
    print("Model restored.")

    for i in xrange(len(data)):
        inp = np.reshape(data[i], (32,32))
        out = cv2.resize(inp, (28,28))
        _,out =  cv2.threshold(out,1,255,cv2.THRESH_BINARY)
        inp = np.reshape(out, (1,784))
        x = tf.stack(inp/255)

        y = tf.nn.softmax(tf.matmul(x,W) + b)
        arr=np.array(sess.run(y))
        print np.argmax(arr),arr.max()
        cv2.imshow("out",out)
        cv2.waitKey(0)
