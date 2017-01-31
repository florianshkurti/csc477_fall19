import scipy
import numpy as np

class LQR(object):
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
    def compute_policy_gains(self, T, dt):
        # Need to stabilize the system around error = 0, command = 0

        if type(self.A) != type([]):
            self.A = T*[self.A] 

        if type(self.B) != type([]):
            self.B = T*[self.B] 
            
        
        self.P = (T+1)*[self.Q]
        self.K = (T+1)*[0]
        
        for t in range(1, T + 1):
            
            self.K[t] = np.dot(self.B[T-t].transpose(), np.dot(self.P[t-1], self.A[T-t]))

            F = self.R + np.dot(self.B[T-t].transpose(), np.dot(self.P[t-1], self.B[T-t]))
            F = np.linalg.inv(F)
            
            self.K[t] = -np.dot(F, self.K[t])
            
            C = self.A[T-t] + np.dot(self.B[T-t], self.K[t])
            E = np.dot(self.K[t].transpose(), np.dot(self.R, self.K[t]))
            
            self.P[t] = self.Q + E + np.dot(C.transpose(), np.dot(self.P[t-1], C))

        
        self.K = self.K[1:]
        self.K = self.K[::-1]
        self.P = self.P[::-1]
        return self.K
