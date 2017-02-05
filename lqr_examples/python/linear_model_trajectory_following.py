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

    p_star_1 = np.array([10 + np.linspace(0,10, 1250), 25.0*np.ones((1,1250))[0] ])
    p_star_2 = np.array([20.0*np.ones((1,1251))[0], 25.0 - np.linspace(0, 10, 1251) ])
    p_star = np.concatenate([p_star_1, p_star_2], axis=1)
    
    # State vector = [x, y, vx, vy]
    x_init = np.array([10, 30, 0, .0], dtype='float64').transpose()
    x_star_init = np.array([p_star[0, 0], p_star[1, 0], 0.0, 0.0])
    z_init = np.concatenate([x_init - x_star_init, np.array([1])])

    A = np.eye(4)
    A[2,2] = (1-dt*friction)/mass
    A[3,3] = (1-dt*friction)/mass
    A[0,2] = dt
    A[1,3] = dt
    
    B = np.array([[0, 0], [0, 0], [dt/mass, 0], [0, dt/mass]], dtype='float64')

    A_bar = []
    for t in range(T):
        x_star_t = np.array([p_star[0, t], p_star[1, t], 0.0, 0.0])
        x_star_tp1 = np.array([p_star[0, t+1], p_star[1, t+1], 0.0, 0.0])
        
        c_t = np.dot(A, x_star_t) - x_star_tp1
        A_bar_t = np.zeros((5,5), dtype='float64')
        A_bar_t[:4, :4] = A
        A_bar_t[:4, 4] = c_t
        A_bar_t[4,4] = 1
        A_bar.append(A_bar_t)

    B_bar = np.zeros((5,2), dtype='float64')
    B_bar[:4, :] = B
    
    Q = np.eye(5)
    R = np.eye(2)    
    
    lqr = LQR(A_bar, B_bar, Q, R)
    K = lqr.compute_policy_gains(T, dt)

    x = x_init
    z = z_init
    X = np.zeros((T, 4), dtype='float64')
    U = np.zeros((T, 2), dtype='float64')
    
    for i in range(T):
        x_star_i = np.array([p_star[0, i], p_star[1, i], 0.0, 0.0])
        z = np.concatenate([x - x_star_i, np.array([1])])
        u = np.dot(K[i], z)

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
    plt.plot( p_star[0, :], p_star[1, :], 'r')
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.legend(['lqr', 'desired'])
    plt.show()


    
