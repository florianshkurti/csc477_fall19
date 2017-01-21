#!/usr/bin/env python
import rospy
import tf
from tf import TransformerROS
from std_srvs.srv import Empty
from std_msgs.msg import String, Header
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PolygonStamped, PointStamped, Point32, Twist
from math import sqrt, cos, sin, pi, atan2
from gazebo_msgs.srv import SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
import numpy
import sys
from tf import transformations as tr

class PID:
    def __init__(self, kp, Td, Ti, dt):
        self.kp = kp
        self.Td = Td
        self.Ti = Ti
        self.curr_error = 0
        self.prev_error = 0
        self.sum_error = 0
        self.prev_error_deriv = 0
        self.curr_error_deriv = 0
        self.control = 0
        self.dt = dt
        
    def update_control(self, err, reset_prev=False):
        self.prev_error = self.curr_error
        self.curr_error = err
        self.sum_error = self.sum_error + err*self.dt
        self.prev_error_deriv = self.curr_error_deriv

        alpha = 0.1
        self.curr_error_deriv = alpha * self.prev_error_deriv + (1-alpha)*(self.curr_error - self.prev_error)/self.dt

        if reset_prev:
            self.curr_error_deriv = 0

        ki = 0
        if abs(self.Ti) > 1e-15:
            ki = 1.0/self.Ti
            
        self.control = self.kp*(self.curr_error + self.Td*self.curr_error_deriv + ki*self.sum_error)
                
        
class StanleyFeedbackControl:
    def __init__(self):
        rospy.init_node('stanley_feedback_control', anonymous=True)

        self.so_seq = 0
        self.max_sensor_range = 8   # TODO: This is set in the kinect urdf. Is there a way to read it as a param?  
        self.occ_grid = None
        self.grid_map = None

        self.clicked_pts_path, self.yaw_path, self.yaw_rate_path = self.compute_target_path()
        
        self.cmd_curr = Twist() 
        self.ref_frame = PoseStamped()
        self.tf_listener = tf.TransformListener()

        self.feedback_control_idx = 0
        self.period = 0.02
        self.angle_pid = PID(1,0.7,0, 1)

        
        self.distance_pid = PID(2.,0.6,0, self.period) 
        self.last_cmd_was_from_single_feedback = False

        self.start_time = None
        self.final_time = None
        
        #self.clicked_pt_sub = rospy.Subscriber("/clicked_point", PointStamped, self.clicked_pt_callback)
        self.path_pub = rospy.Publisher("/simulator/target_path", Path, queue_size=1)
        self.cmd_pub = rospy.Publisher("/husky_1/cmd_vel", Twist, queue_size=1)
        self.ref_frame_pub = rospy.Publisher("/husky_1/ref_path/ref_frame", PoseStamped, queue_size=1)
    
        rospy.wait_for_service('gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/delete_model')
        self.delete_model_srv = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        
        
    def compute_target_path(self):
        clicked_pts_path = Path()
        
        ## Avoid the husky in front of you
        xyz = [(-1.5, 2.0, 0.0), (-1.5, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (4.0, 2.0, 0.0), (4.0, 5.0, 0.0),
               (-5.0, 5.0, 0.0), (-5.0, 2.0, 0.0)]

        ps0 = PoseStamped()
        ps0.header.frame_id = 'world'
        ps0.pose.position = Point(*xyz[0])
        clicked_pts_path.header.frame_id = 'world'
        clicked_pts_path.poses.append(ps0)

        for i in range(1, len(xyz)):
            xc = xyz[i][0]
            yc = xyz[i][1]
            zc = xyz[i][2]

            xp = xyz[i-1][0]
            yp = xyz[i-1][1]
            zp = xyz[i-1][2]

            xlin = numpy.linspace(xp, xc, 100)
            ylin = numpy.linspace(yp, yc, 100)
            zlin = numpy.linspace(zp, zc, 100)
            
            for x,y,z in zip(xlin[1:], ylin[1:], zlin[1:]):
                ps = PoseStamped()
                ps.header.frame_id = 'world'
                ps.pose.position = Point(x,y,z)
                clicked_pts_path.poses.append(ps)
        
        yaw_path = numpy.zeros((1, len(clicked_pts_path.poses)), dtype='float64')
        for i in range(1, len(clicked_pts_path.poses)):
            xc = clicked_pts_path.poses[i].pose.position.x
            yc = clicked_pts_path.poses[i].pose.position.y

            xp = clicked_pts_path.poses[i-1].pose.position.x
            yp = clicked_pts_path.poses[i-1].pose.position.y
            
            dy = yc - yp
            dx = xc - xp
            yaw_path[0, i] = 0.3*yaw_path[0, i-1] + 0.7*atan2(dy, dx);
        
        yaw_rate_path = numpy.diff(yaw_path);

        return (clicked_pts_path, yaw_path, yaw_rate_path)

        
    def clicked_pt_callback(self, msg):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position = msg.point
        self.clicked_pts_path.header = msg.header 
        self.clicked_pts_path.poses.append(ps)
        
   
        
    def get_closest_line_segment(self, xy, path):
        npath = numpy.array([[p.pose.position.x for p in path.poses], [p.pose.position.y for p in path.poses]], dtype='float64')
        dists = numpy.linalg.norm(npath - xy, axis=0)
        
        min_idx = -1
        min_dist = float("inf")
        
        for i in range(len(path.poses)):
            if dists[i] <= min_dist:
                min_dist = dists[i]
                min_idx = i

        assert (min_idx >= 0)

        a = numpy.zeros((2,1), dtype='float64')
        n = numpy.zeros((2,1), dtype='float64')

        k = 15
        if min_idx >= (len(dists)-k):
            a = npath[:, -2]
            n = npath[:, -1] - a
            
        else:
            a = npath[:, min_idx]
            n = npath[:, min_idx + k] - a

        a = numpy.reshape(a, (2,1))
        n = numpy.reshape(n, (2,1))
        
        if numpy.linalg.norm(n) > 1e-15:
            n = n/numpy.linalg.norm(n)
            
        dd = numpy.dot((a-xy).transpose(), n)[0]
        min_dist_to_segment = numpy.linalg.norm((a-xy) - dd*n)
        #a = xy + (a-xy) - dd*n
        #print "a,n,dd,md=", a, n, dd, min_dist_to_segment
        return (a, n, min_dist_to_segment, min_idx)

    
         
    def compute_next_cmd(self):
        
        try:
            follower_frame = '/husky_1/base_link'
            (p_world_follower, q_world_follower) = self.tf_listener.lookupTransform('/world',
                                                                                    follower_frame,
                                                                                    rospy.Time(0))

            xy = numpy.array([p_world_follower[0:2]]).reshape((2,1))
            a, n, min_dist_to_segment, min_idx = self.get_closest_line_segment(xy, self.clicked_pts_path)

            self.ref_frame.header.frame_id = 'world' 
            self.ref_frame.pose.position.x = a[0, 0]
            self.ref_frame.pose.position.y = a[1, 0]
            self.ref_frame.pose.position.z = 0.3
            q = tr.quaternion_from_euler(0, 0, atan2(n[1,0], n[0,0]))
            self.ref_frame.pose.orientation.x = q[0]
            self.ref_frame.pose.orientation.y = q[1]
            self.ref_frame.pose.orientation.z = q[2]
            self.ref_frame.pose.orientation.w = q[3]

            
            if not self.start_time:
                self.start_time = rospy.Time.now()

            if (min_idx >= len(self.clicked_pts_path.poses) - 1):
                self.cmd_prev = self.cmd_curr
                self.cmd_curr = Twist()

                if not self.final_time:
                    self.final_time = rospy.Time.now()
                    
                print "Traversed path in ", (self.final_time - self.start_time).to_sec(), " secs"
                return 
            
            #roll_world_follower, pitch_world_follower, yaw_world_follower = tr.euler_from_quaternion(q_world_follower)
            #target_theta = atan2(n[1, 0], n[0, 0])
            #dtheta = target_theta - yaw_world_follower
            #dtheta = min(dtheta, 2*pi - dtheta)
            #n_perp = numpy.array([ -n[1, 0], n[0, 0] ]).reshape((1,2))
            #signed_md = -min_dist_to_segment * numpy.sign(numpy.dot(n_perp, xy - a)) 

            T_w_ref = tr.quaternion_matrix(q)
            T_w_ref[0,3] = self.ref_frame.pose.position.x
            T_w_ref[1,3] = self.ref_frame.pose.position.y
            T_w_ref[2,3] = self.ref_frame.pose.position.z
            
            T_w_curr = tr.quaternion_matrix(q_world_follower)
            T_w_curr[0,3] = p_world_follower[0]
            T_w_curr[1,3] = p_world_follower[1]
            T_w_curr[2,3] = p_world_follower[2]

            T_curr_ref = numpy.dot(numpy.linalg.inv(T_w_curr), T_w_ref)
            T_ref_curr = numpy.linalg.inv(T_curr_ref)
                        
            droll_curr, dpitch_curr, dyaw_curr = tr.euler_from_quaternion(tr.quaternion_from_matrix(T_curr_ref))
            ddist = numpy.linalg.norm(T_curr_ref[0:2, 3])

            if T_ref_curr[1, 3] > 0:
                ddist = -ddist

            self.cmd_prev = self.cmd_curr
            
            if abs(dyaw_curr) < pi/3.0:
                # robot is facing forwards compared to the ref frame
                self.exec_cascade_feedback_step(ddist, dyaw_curr, self.yaw_rate_path[0, min_idx])
            else:
                # robot is facing backwards compared to the ref frame
                self.exec_single_feedback_step(dyaw_curr)
            
        except Exception as e:
            print e
            self.cmd_curr = Twist()

    def exec_cascade_feedback_step(self, ddist, dyaw_curr, path_yaw_rate):

        kd_yaw = 0.1
        kp_dist = 3.6
        vx = 1
        ksoft = 0.1

        self.cmd_curr = Twist()
        self.cmd_curr.linear.x = vx
        
        if 0: #(self.feedback_control_idx % 5) == 0:
            # regulate angle error
            self.angle_pid.update_control(dyaw_curr)
            self.cmd_curr.angular.z = self.angle_pid.control
            
        else:
            # regulate distance error
            dw = 0 # kd_yaw*(path_yaw_rate - self.cmd_curr.angular.z)
            #self.cmd_curr.angular.z = atan2(kp_dist * ddist, ksoft + vx) - dw
            self.distance_pid.update_control(ddist, self.last_cmd_was_from_single_feedback)
            self.cmd_curr.angular.z = self.distance_pid.control

            
        self.feedback_control_idx = self.feedback_control_idx + 1
        self.last_cmd_was_from_single_feedback = False

    def exec_single_feedback_step(self, dyaw_curr):
        self.cmd_curr = Twist()
        self.cmd_curr.linear.x = 0
        self.cmd_curr.angular.z = dyaw_curr
        self.last_cmd_was_from_single_feedback = True
            
    def run(self):
        rate = rospy.Rate(1.0/self.period)
        while not rospy.is_shutdown():
            self.compute_next_cmd()
            self.cmd_pub.publish(self.cmd_curr)
            self.ref_frame_pub.publish(self.ref_frame)
            self.path_pub.publish(self.clicked_pts_path)
            rate.sleep()

    
if __name__ == '__main__':
    sfc = StanleyFeedbackControl()
    sfc.run()


