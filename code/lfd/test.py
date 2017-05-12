# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:32:36 2017

@author: dsbrown
"""
import numpy as np
import matplotlib.pyplot as plt

filename = "/home/dsbrown/Code/gsim/data/user_0/experiment_0/motion.npy"
data = np.load(filename) #gives data for all trajectories!
rep = 0
plt.plot(data[rep][:,0],data[rep][:,1])
print "init", data[rep][0]
print "final", data[rep][-1]
