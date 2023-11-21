#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import inv


# In[2]:


def tellme(s):
    plt.title(s, fontsize=16)
    fig = plt.gcf()
    fig.set_size_inches(18,10,forward=True)
    plt.draw()


def point_picker():
    plt.clf()
    plt.setp(plt.gca(), autoscale_on=False)
#     plt.waitforbuttonpress()
    tellme('Pick some points to fit the line, click to begin')
    plt.waitforbuttonpress()

    while True:
        pts = []
        tellme('Select points with left mouse, stop by clicking right mouse')
        pts = np.asarray(plt.ginput(n=-1, timeout=0,mouse_pop=2,mouse_stop=3))
        plt.plot(pts[:,0],pts[:,1], 'kx')
        return pts
    
def my_linfit(x,y):
    W = [[np.sum(x**2), np.sum(x)]
        ,[np.sum(x), len(x)]]
    R = [[np.sum(x*y)]
        ,[np.sum(y)]] 
    X = np.array(inv(W).dot(R))
    return X[0,:],X[1,:] 


# In[3]:


while True:
    training_set = point_picker()
    a,b = my_linfit(training_set[:,0],training_set[:,1])
    xp = np.arange(-2,5,0.1)
    plt.plot(xp,a*xp+b,'r-')
    
    print(f"My fit: a={a} and b={b}")
    plt.draw()
    
    plt.title("Left click to continue, press any keyboard to stop", fontsize=16)
    if plt.waitforbuttonpress():
        break 
    


# In[ ]:



