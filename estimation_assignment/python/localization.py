#!/usr/bin/env python
import rospy
import tf
import tf.transformations as tr
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from math import sqrt, cos, sin, pi, atan2
from threading import Thread, Lock
from math import pi, log, exp
import random
import numpy as np
from scipy.optimize import least_squares
import sys


class Landmark(object):
    dim = 2
    def __init__(self, landmark_id, x, y):
        self.id = landmark_id
        self.x = float(x)
        self.y = float(y)

    def as_vector(self):
        return np.array([self.x, self.y], dtype='float64').transpose()

    def diff(self, other_landmark):
        return self.as_vector() - other_landmark.as_vector()
    
class State(object):
    dim = 2
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def as_vector(self):
        return np.array([self.x, self.y], dtype='float64').transpose()

    def diff(self, other_state):
        return np.array([self.x - other_state.x, self.y - other_state.y], dtype='float64').transpose()

    def add_noise(self, noise):
        self.x += noise[0]
        self.y += noise[1]
        
    
class Control(object):
    dim = 2
    def __init__(self, vx, vy):
        self.vx = float(vx)
        self.vy = float(vy)

        
class Observation(object):
    dim = 1
    def __init__(self, landmark_id, rng):
        self.landmark_id = landmark_id
        self.rng = float(rng)

    def add_noise(self, noise):
        self.rng += noise[0]
        
    def as_vector(self):
        return np.array([self.rng]).transpose()

    def diff(self, other_observation):
        return self.as_vector() - other_observation.as_vector()



    
def dynamics_model(state, control, dt):
    # TODO: fill this
    #expected_state = State(?, ?)
    return expected_state

def measurement_model(state, landmark):
    # TODO: fill this
    #expected_observation = Observation(?, ?)
    return expected_observation
    
    
class LocalizationWithRangeMeasurements(object):
    def __init__(self,
                 init_state,
                 landmarks,
                 observations_across_time,
                 controls_across_time,
                 num_timesteps,
                 dt,
                 dynamics_noise_std_dev,
                 obs_noise_std_dev):
        
        self.observations_across_time = observations_across_time
        self.controls_across_time = controls_across_time
        self.init_state = init_state
        self.landmarks = landmarks
        self.num_timesteps = num_timesteps
        self.dt = dt
        self.dynamics_noise_std_dev = dynamics_model_std_dev
        self.obs_noise_std_dev = obs_noise_std_dev
        
    def dynamics_cost(self, state_curr, state_prev, control_prev):
        #TODO: evaluate the dynamics model and return the difference between
        #the current state and expected state as predicted from the dynamics 
        # I.e. returns a vector 
        return diff

    
    def measurement_cost(self, state, landmark, observation):
        # TODO: evaluate the observation model and return the difference
        # between the given observation and the one predicted by the
        # measurement model
        # I.e. returns a vector
        return diff

    
    def cost_function(self, x):

        # TODO: Extend the vector F, so that it contains all the residuals
        # that you are going to use as errors in your least squares formulation
        # Note: F = np.concatenate((F, another_vector)) is the way to extend it
        F = np.array([])
        
        T = self.num_timesteps


        # TODO: Extend F by the deviation of the estimated first state (x[0], x[1])
        # from the given first state self.init_state
        # deviation_from_init_state = ?
        F = np.concatenate((F, deviation_from_init_state))

        
        for t in xrange(1, T):
            #TODO: Extend F by the dynamical model error from one estimated state to the next
            #      as defined by the current vector x 
            F = np.concatenate((F, dc))
        
        for t in xrange(self.num_timesteps):
            #TODO: Extend F by the observation model error of the observations made at time t
            #      in estimated state at time t, as currently defined by vector x  
            
        return F
        
    def localize(self):
        T = self.num_timesteps
        
        x_init = np.zeros((State.dim * T, ), dtype='float64')
        res = least_squares(self.cost_function, x_init)
        
        resulting_state_estimates_across_time = T * [ 0 ]
        for t in xrange(T):
            resulting_state_estimates_across_time[t] = State(res.x[State.dim*t], res.x[State.dim*t+1])
            
        return resulting_state_estimates_across_time

    
if __name__ == "__main__":

    # Populating landmarks
    landmarks = []
    landmarks.append(Landmark(0,  1,-15))
    landmarks.append(Landmark(1,  -15,5))
    landmarks.append(Landmark(2,  -15,20))
    landmarks.append(Landmark(3,  1,7))
    landmarks.append(Landmark(4,  -1,20))
    landmarks.append(Landmark(5,  -10,-13))
    
    num_landmarks = len(landmarks)
        
    dt = 1
    num_timesteps = 10

    # We populate a list of num_timesteps controls, which are known
    controls = []
    controls.append(Control(-10, 5))
    controls.append(Control(1, 2))
    controls.append(Control(1, 4))
    controls.append(Control(-3, 0))
    controls.append(Control(-1, -2))
    controls.append(Control(-1, -2))
    controls.append(Control(-3, 3))
    controls.append(Control(-3, 5))
    controls.append(Control(-3, -7))


    # Noise added to the dynamics of the system
    dynamics_noise_std_dev = 0.5
    dynamics_noise_covariance = (dynamics_noise_std_dev**2) * np.identity(State.dim)
    dynamics_noise_mean = np.zeros((State.dim, ))

    # These controls make the system move and visit num_timesteps States
    states = [State(0,0)]
    for t in xrange(1, num_timesteps):
        next_state = dynamics_model(states[t-1], controls[t-1], dt)
        next_state.add_noise(np.random.multivariate_normal(dynamics_noise_mean, dynamics_noise_covariance))
        states.append(next_state)
        
        
    
    # Observations across time
    observation_indexes_across_time = num_timesteps * [ 0 ]
    observation_indexes_across_time[0] = [2,3]
    observation_indexes_across_time[1] = [0,1,4,5]
    observation_indexes_across_time[2] = [0,3,4,5]
    observation_indexes_across_time[3] = [2,3,5]
    observation_indexes_across_time[4] = [0,1,5]
    observation_indexes_across_time[5] = [0,1,3,5]
    observation_indexes_across_time[6] = [4,5]
    observation_indexes_across_time[7] = [2,3]
    observation_indexes_across_time[8] = [4,5]
    observation_indexes_across_time[9] = [0,1]

    # Noise added to the observations of the system
    obs_noise_std_dev = 0.5
    obs_noise_covariance = (obs_noise_std_dev**2) * np.identity(Observation.dim)
    obs_noise_mean = np.zeros((Observation.dim, ))


    # Creates observation objects according to the indexes specified above
    observations_across_time = num_timesteps * [ 0 ]
    for t in xrange(num_timesteps):
        observations_across_time[t] = []
        for i in observation_indexes_across_time[t]:
            z_t_i = measurement_model(states[t], landmarks[i])
            z_t_i.add_noise(np.random.multivariate_normal(obs_noise_mean, obs_noise_covariance))
            observations_across_time[t].append(z_t_i)

    
    solver = LocalizationWithRangeMeasurements(states[0],
                                               landmarks,
                                               observations_across_time,
                                               controls,
                                               num_timesteps,
                                               dt,
                                               dynamics_noise_std_dev,
                                               obs_noise_std_dev)
    
    resulting_states = solver.localize()

    print "Norm of difference between estimated states and true states"
    for t in xrange(num_timesteps):
        print np.linalg.norm(resulting_states[t].diff(states[t]))
        
    true_rx = [r.x for r in states]
    true_ry = [r.y for r in states]
    estimated_rx = [r.x for r in resulting_states]
    estimated_ry = [r.y for r in resulting_states]

    true_lx = [l.x for l in landmarks]
    true_ly = [l.y for l in landmarks]
    
    
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt

    
    est_robot_states, = plt.plot(estimated_rx, estimated_ry, 'bo-', label="Estimated states")
    true_robot_states, = plt.plot(true_rx, true_ry, 'ro-', label="True states")
    plt.plot(true_lx, true_ly, 'y*', markersize=15)
    plt.legend(handles=[est_robot_states, true_robot_states])
    plt.ylim([-30, 30])
    
    for l in landmarks:
        plt.annotate(str(l.id),
                     xy=(l.x, l.y),
                     xytext=(-20, 20),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    for t in xrange(num_timesteps):
        for obs in observations_across_time[t]:
            l = landmarks[obs.landmark_id]
            plt.plot([l.x, states[t].x], [l.y, states[t].y], 'k--', alpha=0.3)
        
    plt.show()
    
