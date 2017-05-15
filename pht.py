import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import input_data
import cv2

ks=30
ker = np.ones((ks,ks))/ks/ks
img=cv2.imread('1.jpeg')
gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
sh= gray.shape
gray=cv2.resize(gray,(sh[0]/5,sh[1]/5))
sh= gray.shape
cropy=60
cropx=30
gray=gray[cropx:sh[0]-cropx,cropy:sh[1]-cropy]
gray=255-(gray-np.min(gray))*(255/np.max(gray))
gray=gray-np.mean(gray)
#gray=cv2.filter2D(gray.astype(float),-1,ker)
gray = cv2.GaussianBlur(gray,(9,9),0)
sh= gray.shape
k=0
loc=np.zeros((1,2))
for i in range(ks,sh[0]-ks):
    for j in range(ks,sh[1]-ks):
        if((gray[i,j]>gray[i-1,j]) and (gray[i,j]>gray[i-1,j-1]) and gray[i,j]>gray[i,j-1]):
            if((gray[i,j]>gray[i+1,j]) and (gray[i,j]>gray[i+1,j+1]) and gray[i,j]>gray[i,j+1]):
                if((gray[i,j]>gray[i+1,j-1]) and (gray[i,j]>gray[i-1,j+1])):
                    k=k+1
                    loc=np.vstack((loc,np.array([i,j])))
loc=loc.astype(int)
print loc

for i in xrange(k):
    gray=cv2.circle(gray.astype(np.uint8), (loc[i+1,0],loc[i+1,1]), 10,255)
cv2.imshow("gray",gray)
cv2.waitKey(0)
