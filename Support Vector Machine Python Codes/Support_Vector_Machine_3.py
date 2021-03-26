#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:24:27 2021

@author: batuhan
"""
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X,y=make_blobs(centers=2,random_state=0,cluster_std=0.7)
model = SVC(kernel='linear', C=1050)
model.fit(X, y)


def svc(model,ax=None):   
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x=np.linspace(xlim[0],xlim[1])
    y=np.linspace(ylim[0],ylim[1])
    Y,X= np.meshgrid(y,x)
    xy=np.vstack([X.ravel(),Y.ravel()]).T
    P=model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X,Y,P,colors="r",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    

def svm(N=20, ax=None):
 X, y = make_blobs( centers=2,
   random_state=0, cluster_std=0.70)
 X = X[:N]
 y = y[:N]
 model = SVC(kernel='linear', C=18)
 model.fit(X, y)
 ax = ax or plt.gca()
 ax.scatter(X[:, 0], X[:, 1], c=y, s=80, cmap='Accent')
 ax.set_xlim(-1, 4)
 ax.set_ylim(-1, 7)
 svc(model, ax)
 
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
for axi, N in zip(ax, [40, 55]):
 svm(N, axi)
 axi.set_title('Veri = {0}'.format(N))

plt.show()