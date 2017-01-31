#!/usr/bin/python
from math import sin, cos, pi

import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
from lqr import LQR 
        

if __name__ == "__main__":

    T = 2500
    dt = 0.01
    mass = 1.0
    friction = 0.1

    # State vector = [x, y, vx, vy]
    x_init = np.array([10, 30, 10, -5.0], dtype='float64').transpose()
    
    A = np.eye(4)
    A[2,2] = (1-dt*friction)/mass
    A[3,3] = (1-dt*friction)/mass
    A[0,2] = dt
    A[1,3] = dt
    
    B = np.array([[0, 0], [0, 0], [dt/mass, 0], [0, dt/mass]], dtype='float64')
    
    Q = 0.01*np.eye(4)
    R = np.eye(2)
    
    lqr = LQR(A,B,Q,R)
    K = lqr.compute_policy_gains(T, dt)

    x = x_init
    X = np.zeros((T, 4), dtype='float64')
    U = np.zeros((T, 2), dtype='float64')
    
    for i in range(T):
        u = np.dot(K[i], x)

        # This is essentially the simulator of the vehicle
        x = np.dot(A, x) + np.dot(B, u)
        
        
        X[i, :] = x.transpose()
        U[i, :] = u.transpose()


    plt.switch_backend('Qt4Agg')
    
    plt.figure()
    plt.plot( X[:, 0], '-b')
    plt.plot( X[:, 1], '-r')
    plt.plot( X[:, 2], '-g')
    plt.plot( X[:, 3], '-k')
    plt.legend(['x', 'y', 'vx', 'vy'])
    plt.xlabel('time steps')
    plt.show()
    
    
    plt.figure()
    plt.plot( U[:, 0], 'b')
    plt.plot( U[:, 1], 'r')
    plt.legend(['ux', 'uy'])
    plt.xlabel('time steps')
    plt.ylabel('controls')
    plt.show()

    plt.figure()
    plt.plot( X[:, 0], X[:, 1], 'b')
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    
    plt.show()


    
