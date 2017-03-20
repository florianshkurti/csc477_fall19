#!/usr/bin/python
from kalman_filter import KalmanFilter
import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')
    
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def plot_mean_and_covariance(mean, Sigma):
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    # this contour of the ellipse is going to contain about 50% of the probability mass
    #alpha = Sigma.shape[0]

    # this contour of the ellipse is going to contain about 90% of the probability mass
    alpha = Sigma.shape[0] + 2*np.sqrt(Sigma.shape[0])
    
    # See http://stanford.edu/class/ee363/lectures/estim.pdf for more info on this

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ell = Ellipse(xy=(mean[0, 0], mean[1, 0]),
                  width=np.sqrt(alpha*eigenvalues[0]),
                  height=np.sqrt(alpha*eigenvalues[1]),
                  angle=np.rad2deg(np.arccos(eigenvectors[0, 0])))

    ell.set_facecolor('none')
    ax.add_artist(ell)
    plt.xlim([-10, 20])
    plt.ylim([-30, 30])
    plt.show()
    
if __name__ == "__main__":

    
    # Discrete dynamics:
    #           px_{t+1} = px_t + delta_t*vx_t + 1/2*delta_t^2*w_x(t)
    #           py_{t+1} = py_t + delta_t*vy_t + 1/2*delta_t^2*w_y(t) - 1/2*g*delta_t^2
    #           vx_{t+1} = vx_t + delta_t*w_x(t) 
    #           vy_{t+1} = vy_t + delta_t*w_y(t) -g*delta_t
    #
    # Define:  
    #           x_t = [px_t; py_t; vx_t; vy_t]    (4x1 vector)
    #           w_t = [w_x(t); w_y(t)]   where w_t ~ N([0; 0], sigma_w^2 * I)
    # 
    #           x_{t+1} = Ax_t + Bu_t + Gw_t
    #
    #           A = [1  0 delta_t  0; 0  1 0 delta_t; 0 0 1 0; 0 0 0 1]
    #           B = I
    #           u_t = [0; -1/2g*delta_t^2; 0; -g*delta_t]
    #           G = [1/2*delta_t^2  0; 0  1/2*delta_t^2; delta_t 0; 0 delta_t]
    #            
    #
    # Suppose we measure position with noise. Then:
    #
    #           z_t = [1 0 0 0; 0 1 0 0] * x_t + n_t  where n_t ~ N([0; 0], sigma_n^2 * I) 
    #
    
    delta_t = 0.1 # predictions are going to be made for 0.5 seconds in the future
    g = 9.81
    
    A = np.array([[1, 0, delta_t, 0], [0,  1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.identity(4)
    G = np.array([[delta_t**2/2, 0], [0, delta_t**2/2], [delta_t, 0], [0, delta_t]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    sigma_w = 0.1  # standard deviation of position prediction noise for one time step of dt seconds
    Q = (sigma_w**2) * np.identity(2)

    sigma_n = 0.2  # standard deviation of position measurement noise
    R = (sigma_n**2) * np.identity(2)

    sigma_p = 3 # uncertainty about initial position
    sigma_v = 3  # uncertainty about initial velocity
    Sigma_init = np.array([[sigma_p**2, 0, 0, 0],
                           [0, sigma_p**2, 0, 0],
                           [0, 0, sigma_v**2, 0],
                           [0, 0, 0, sigma_v**2]])

    x_init = np.array([[0, 2, 2, 5]]).transpose() 
    kf = KalmanFilter(A, B, G, H, Q, R, x_init, Sigma_init)
    u = np.array([[0, -0.5*g*delta_t**2, 0, -g*delta_t]]).transpose()
    
    x_estimated = [x_init]
    cov_estimated = [Sigma_init]

    x_real_list = [np.array([[0, 0, 5, 10]]).transpose()]
    x_real = x_real_list[0]
    z_list = []
    
    for i in xrange(100):

        kf.predict(u)

        # simulate noisy observation
        x_real = A.dot(x_real)+u
        z = x_real[0:2, :] + np.random.multivariate_normal(np.zeros((2,)), R).reshape((2,1))

        kf.update(z)

        x_estimated.append(kf.x)
        cov_estimated.append(kf.Sigma)
        x_real_list.append(x_real)
        z_list.append(z)

    plt.figure()
    px = np.array([ pv[0, 0] for pv in x_real_list])
    py = np.array([ pv[1, 0] for pv in x_real_list])
    plt.plot(px, py, 'r', label="True position")


    gx = np.array([ pv[0, 0] for pv in x_estimated])
    gy = np.array([ pv[1, 0] for pv in x_estimated])
    plt.plot(gx, gy, 'b', label="KF position estimate")

    sy = [np.sqrt(Sigma[1,1]) for Sigma in cov_estimated]
    yerr_plus = [y + 3*yerr for y, yerr in zip(gy, sy)]
    yerr_minus = [y - 3*yerr for y, yerr in zip(gy, sy)]
    
    plt.fill_between(gx, yerr_minus, yerr_plus, facecolor='b', alpha=0.2, edgecolor='none', label="")

    zx = np.array([ z[0, 0] for z in z_list])
    zy = np.array([ z[1, 0] for z in z_list])
    plt.plot(zx, zy, 'gx--', label="Noisy measurements")
    
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xlabel('px')
    plt.ylabel('py')
    plt.legend()


    plt.figure()
    vy = np.array([ pv[3, 0] for pv in x_real_list])
    t = range(len(x_real_list))
    plt.plot(t, vy, 'r', label="True y-velocity")
    
    evy = np.array([ pv[3, 0] for pv in x_estimated])
    plt.plot(t, evy, 'b', label="KF y-velocity estimate")

    sy = [np.sqrt(Sigma[3,3]) for Sigma in cov_estimated]
    yerr_plus = [y + 3*yerr for y, yerr in zip(evy, sy)]
    yerr_minus = [y - 3*yerr for y, yerr in zip(evy, sy)]
    
    plt.fill_between(t, yerr_minus, yerr_plus, facecolor='b', alpha=0.2, edgecolor='none', label="")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.xlabel('time step')
    plt.ylabel('vy')
    plt.legend()

    
    plt.show()
