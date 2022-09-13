# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 16:07:20 2022

@author: Harish
Some code is borrowed from 
https://francescoturci.net/2020/06/19/minimal-vicsek-model-in-python/
"""

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 

def generate_vicsek_data(iterations, N, eta, v, r, L=1):
    """
    Assuming a square periodic domain of size 1
    N: Total number of particles
    iters: Number of iterations to calculate the positions and orientations:
    eta: noise in alignment
    r: interaction radius
    v: velocity magnitude
    L: side length of the square domain
    
    Output:
        3 arrays of length equal to iters. They are x position, y position and orientation
    """
    #global orient
    
    x_pos = np.zeros([iterations, N])
    y_pos = np.zeros([iterations, N])
    orient_ = np.zeros([iterations, N])
    
    #randomly assigning initial positions and orientations
    x_pos[0] = np.random.rand(N)*L
    y_pos[0] = np.random.rand(N)*L
    orient_[0] = np.random.rand(N)*2*np.pi-np.pi
    global orient
    orient = np.random.rand(N)*2*np.pi-np.pi

    pos = np.column_stack([x_pos[0], y_pos[0]])
    for i in range(iterations-1):
        """
        the next few lines involves calculation of particles which are in a 
        neighbourhood of certain interaction radius for each particle. This is 
        better than brute force calculation of distances to every particles. 
        Time for brute forcing is proportional to square of number of particles.
        The below code is faster. However, a much faster algorithm uses a data
        structure called as cell list. In the cell list method, the domain is 
        divided into cells(like squares on a chess board) and the cell that a 
        particle belongs is tracked. The cell dimensions are bigger than r 
        but smaller than L. This way we compare distances with particles only 
        in a cell a particle belongs to. This has been omitted as the code can 
        be considerably longer
        

        pos = np.column_stack([x_pos[i], y_pos[i]])
        print("pos", pos.shape)
        tree = cKDTree(pos, boxsize=[L,L])
        dist = tree.sparse_distance_matrix(tree, max_distance=r,output_type='coo_matrix')
        #print(np.max(orient[i+1]))
        #important 3 lines: we evaluate a quantity for every column j
        data = np.exp(orient[i][dist.col]*1j)
        print("data", data.shape)
        # construct  a new sparse marix with entries in the same places ij of the dist matrix
        neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
        # and sum along the columns (sum over j)
        print("neigh", neigh.shape)
        S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
        print(S.shape)
        #print(S)
        orient[i+1,:] = orient[i,:] + np.angle(S) + eta*np.random.uniform(-np.pi, np.pi, size=N)
        #Updating positions
        x_pos[i+1, :] = x_pos[i, :] + v*np.cos(orient[i+1, :])
        y_pos[i+1, :] = y_pos[i, :] + v*np.sin(orient[i+1, :])
    
        #adjusting for periodic BC
        x_pos[i+1][x_pos[i+1]>L] -= L
        y_pos[i+1][y_pos[i+1]>L] -= L
        x_pos[i+1][x_pos[i+1]<0] += L
        y_pos[i+1][y_pos[i+1]<0] += L
        """
        #if (i%100==0):
        #    print(i)
        #print("pos", pos.shape)
        tree = cKDTree(pos,boxsize=[L,L])
        dist = tree.sparse_distance_matrix(tree, max_distance=r,output_type='coo_matrix')
        
        #print(np.max(orient))
        #important 3 lines: we evaluate a quantity for every column j
        data = np.exp(orient[dist.col]*1j)
        #print(data.shape)
        #print("data", data.shape)
        # construct  a new sparse marix with entries in the same places ij of the dist matrix
        neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
        #print("neigh", neigh.shape)
        # and sum along the columns (sum over j)
        S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
        #print("neigh", S.shape)
         
        orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)
     
     
        cos, sin= np.cos(orient), np.sin(orient)
        pos[:,0] += cos*v
        pos[:,1] += sin*v
     
        pos[pos>L] -= L
        pos[pos<0] += L
     
        #qv.set_offsets(pos)
        #qv.set_UVC(cos, sin,orient)
        x_pos[i+1] = pos[:, 0]
        y_pos[i+1] = pos[:, 1]
        orient_[i+1] = orient
        
        
    return x_pos, y_pos, orient_

def animate_viscek(x_pos, y_pos, orient, iterations):
    x_dir = np.cos(orient)
    y_dir = np.sin(orient)
    #print(x_dir.shape)
    #print(x_dir[:,1].shape)
    
    fig, ax= plt.subplots(figsize=(6,6))
    qv = ax.quiver(x_pos[0], y_pos[0], np.cos(orient[0]), np.sin(orient[0]), orient[0], 
                   clim=[-np.pi, np.pi], cmap='hsv')
    
    def animate(i):
        if (i%100==0):
            print(i)
        #print(np.sum(x_dir[i]))
        #print(np.sum(y_dir[i]))
        #print(np.std(orient[i]))
        #plt.hist(orient[i])
        pos = np.column_stack([x_pos[i], y_pos[i]])
        qv.set_offsets(pos)
        qv.set_UVC(x_dir[i, :], y_dir[i, :], orient[i, :])
        return qv,
    
    global animator #need to assign as a global variable
    animator = FuncAnimation(fig, animate, frames=np.arange(1, iterations), interval=0.01)#, blit=True)
    plt.show()
    
