import numpy as np

class KalmanFilter(object):
    def __init__(self, A, B, G, H, Q, R, x_init, Sigma_init):
        self.x = x_init
        self.Sigma = Sigma_init
        self.A = A
        self.B = B
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.B.shape[1], ))
       
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.Sigma = self.A.dot(self.Sigma).dot(self.A.transpose()) + self.G.dot(self.Q).dot(self.G.transpose())
        
    def update(self, z):
        expected_z = self.H.dot(self.x)
        measurement_residual = z - expected_z
        
        Sigma_measurement_residual = self.H.dot(self.Sigma).dot(self.H.transpose()) + self.R

        Kalman_gain = self.Sigma.dot(self.H.transpose()).dot(np.linalg.inv(Sigma_measurement_residual)) 

        self.x = self.x + Kalman_gain.dot(measurement_residual)

        rows = self.Sigma.shape[0]
        self.Sigma = (np.identity(rows) - Kalman_gain.dot(self.H)).dot(self.Sigma)

        
