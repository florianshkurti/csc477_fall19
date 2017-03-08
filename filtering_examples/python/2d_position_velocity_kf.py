#!/usr/bin/python
from kalman_filter import KalmanFilter
import numpy as np

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
    #           p_{t+1} = p_t + delta_t*v_t + w_x(t)
    #           v_{t+1} = v_t + w_y(t)
    #
    # Define:  
    #           x_t = [p_t; v_t]    (2x1 vector)
    #           w_t = [w_x(t); w_y(t)]   where w_t ~ N([0; 0], sigma_w^2 * I)
    # 
    #           x_{t+1} = [1  delta_t; 0  1]*x_t + w_t
    #
    # Suppose we measure position with noise. Then:
    #
    #           z_t = [1 0] * x_t + n_t  where n_t ~ N([0; 0], sigma_n^2 * I) 
    #
    
    delta_t = 0.5 # predictions are going to be made for 0.5 seconds in the future

    A = np.array([[1, delta_t], [0,  1]])
    B = np.zeros((2, 2))
    G = np.identity(2)
    H = np.array([[1, 0]])

    sigma_w = 0.1  # 10cm = standard deviation of position prediction noise for one time step of dt seconds
    Q = (sigma_w**2) * np.identity(2)

    sigma_n = 0.2  # 20cm = standard deviation of position measurement noise
    R = (sigma_n**2) * np.identity(1)

    x_init = np.array([[0, 5]]).transpose()

    sigma_p = 1 # 10cm = uncertainty about initial position
    sigma_v = 20  # 20m/s = uncertainty about initial velocity
    Sigma_init = np.array([[sigma_p**2, 0], [0, sigma_v**2]])
    
    kf = KalmanFilter(A, B, G, H, Q, R, x_init, Sigma_init)

    plot_mean_and_covariance(kf.x, kf.Sigma)

    kf.predict()

    plot_mean_and_covariance(kf.x, kf.Sigma)

    kf.update(z=5)
    
    plot_mean_and_covariance(kf.x, kf.Sigma)
