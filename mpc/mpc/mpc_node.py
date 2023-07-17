#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from utils import nearest_point

from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R
# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 24  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float =  0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 1.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 1.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')

        filename = "sim_points.csv" 
        self.yaw_data = np.load('data.npz')

        # ROS subscribers and publishers
        # use the MPC as a tracker (similar to pure pursuit)
        self.pose_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 1)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, 'drive',1)
        self.goal_points_publisher = self.create_publisher(MarkerArray, 'pp_goal_points',1)
        self.spline_publisher = self.create_publisher(Marker, 'pp_spline',1)
        self.mpc_path_publisher = self.create_publisher(Marker, 'mpc_spline',1)
        self.pp_goal_publisher = self.create_publisher(Marker, 'pp_goal_point',1)

        # TODO: get waypoints here
        self.waypoints = load_points(filename, scaler=1)
        spline_data, uout = splprep(self.waypoints.T, s=0, per=True)
        self.x_spline, self.y_spline = splev(np.linspace(0,1,1000), spline_data)
        self.vx      , self.vy       = splev(np.linspace(0,1,1000), spline_data, der=1)

        self.spline_velocity = (self.vx**2 + self.vy**2)**0.5
        # print(len(self.x_spline))
        self.pp_points_data = self.visualize_pp_goal_points()
        self.pp_spline_data = self.visualize_spline()

        #Publish Rviz Markers every 2 seconds
        self.timer = self.create_timer(2, self.publish_rviz_data)#Publish waypoints


        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

    def pose_callback(self, pose_msg):
        # TODO: extract pose from ROS msg
        # Find the current waypoint to track using methods mentioned in lecture
        current_position = pose_msg.pose.pose.position
        current_quat = pose_msg.pose.pose.orientation

        current_lin_vel = pose_msg.twist.twist.linear
        current_ang_vel = pose_msg.twist.twist.angular

        quaternion = [current_quat.x, current_quat.y, current_quat.z, current_quat.w]
        euler = (R.from_quat(quaternion)).as_euler('xyz', degrees=False)
        global_car_position = [current_position.x, current_position.y, current_position.z] # Current location of car in world frame

        # Calculate immediate goal on Pure Pursuit Trajectory
        spline_points = np.hstack((self.x_spline.reshape(-1,1), self.y_spline.reshape(-1,1), np.zeros_like(self.y_spline.reshape(-1,1))))



        # individual_rays = np.diff(spline_points)
        individual_rays = np.diff(np.roll(spline_points,5)-spline_points)


        ######### Calculate yaw at each point -->
        # unit_rays = individual_rays/np.reshape(np.linalg.norm(individual_rays,axis =1),(-1,1))
        # yaw_array = np.arccos(np.clip(np.dot(unit_rays, [1.0,0.0]), a_min=-1, a_max=1))
        ######### OR method 2 ->
        yaw_array = self.yaw_data['yaw'].flatten()

        # Calculate closest point on spline
        norm_array = np.linalg.norm(spline_points - global_car_position, axis = 1)
        closest_pt_idx = np.argmin(norm_array)
        self.visualize_pt(spline_points[closest_pt_idx])

        # Check if car is oriented opposite the spline array direction
        if(closest_pt_idx+10>(len(self.x_spline)-1)): idx = 10 
        else: idx =  closest_pt_idx+10
        sample_point = global_2_local(quaternion, spline_points[idx], global_car_position)
        if sample_point[0]>0:
            arangeit = np.arange(len(self.x_spline))
            rollit = np.roll(arangeit, -closest_pt_idx)
            # print(rollit)
        else:
            arangeit = np.flip(np.arange(len(self.x_spline)))
            rollit = np.roll(arangeit, closest_pt_idx)
        
        spline_points = spline_points[rollit]
        self.spline_velocity = self.spline_velocity[rollit]
        yaw_array = yaw_array[rollit]

        vehicle_state = State()
        # print(vehicle_state.x)
        vehicle_state.x = current_position.x
        vehicle_state.y = current_position.y
        vehicle_state.yaw = euler[-1]
        if vehicle_state.yaw<0:
            vehicle_state.yaw = vehicle_state.yaw + 2*np.pi
        vehicle_state.v = (current_lin_vel.x**2+current_lin_vel.y**2+current_lin_vel.z**2)**0.5

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_x   = spline_points[:,0]
        ref_y   = spline_points[:,1]
        ref_yaw = yaw_array
        print(vehicle_state.y)
        angle_flip_idx = np.where(ref_yaw<0)
        ref_yaw[angle_flip_idx] = ref_yaw[angle_flip_idx] + 2*np.pi
        print("ref_yaw[0] in deg: ",np.rad2deg(ref_yaw[0]))
        print("actual yaw in deg: ", np.rad2deg(euler[-1]))
        print(" ")
        ref_v   = self.spline_velocity
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            self.ox,
            self.oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)
        self.mpc_spline_data = self.visualize_mpc_path()

        # TODO: publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        msg = AckermannDriveStamped()
        # msg.drive.speed = 0.0
        msg.drive.speed = speed_output
        msg.drive.steering_angle = float(steer_output)
        self.drive_publisher.publish(msg)

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK)) # (16,16)

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))
        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block)) # Shape (36, 36)

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere
        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        constraints += [cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]

        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints += [cvxpy.abs(cvxpy.diff(self.uk[1, :]))/self.config.DTK<=self.config.MAX_DSTEER]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints += [self.xk[:,0]==self.x0k]
        
        constraints += [self.xk[2,:]>=self.config.MIN_SPEED]
        constraints += [self.xk[2,:]<=self.config.MAX_SPEED]

        constraints += [cvxpy.abs(self.uk[0,:])<=self.config.MAX_ACCEL]
        constraints += [cvxpy.abs(self.uk[1,:])<=self.config.MAX_STEER]
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC control with updating operational point iteratively
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    ########################### VISUALIZATION ############################
    def visualize_pt(self, point):
        array_values=MarkerArray()

        message = Marker()
        message.header.frame_id="map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.type= Marker.SPHERE
        message.action = Marker.ADD
        message.id=0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0
        message.scale.x=0.2
        message.scale.y=0.2
        message.scale.z=0.2
        message.color.a=1.0
        message.color.r=1.0
        message.color.b=1.0
        message.color.g=0.0
        message.pose.position.x=float(point[0])
        message.pose.position.y=float(point[1])
        message.pose.position.z=0.0
        message.lifetime.nanosec=int(1e8)

        array_values.markers.append(message)
        self.pp_goal_publisher.publish(message)
    
    def visualize_pp_goal_points(self):
        array_values=MarkerArray()

        for i in range(len(self.waypoints)):
            message = Marker()
            message.header.frame_id="map"
            message.header.stamp = self.get_clock().now().to_msg()
            message.type= Marker.SPHERE
            message.action = Marker.ADD
            message.id=i
            message.pose.orientation.x=0.0
            message.pose.orientation.y=0.0
            message.pose.orientation.z=0.0
            message.pose.orientation.w=1.0
            message.scale.x=0.2
            message.scale.y=0.2
            message.scale.z=0.2
            message.color.a=1.0
            message.color.r=1.0
            message.color.b=0.0
            message.color.g=0.0
            message.pose.position.x=float(self.waypoints[i,0])
            message.pose.position.y=float(self.waypoints[i,1])
            message.pose.position.z=0.0
            array_values.markers.append(message)
        return array_values
    
    def visualize_spline(self):

        message = Marker()
        message.header.frame_id="map"
        message.type= Marker.LINE_STRIP
        message.action = Marker.ADD
        message.pose.position.x= 0.0
        message.pose.position.y= 0.0
        message.pose.position.z=0.0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0
        message.scale.x=0.05

        for i in range(len(self.x_spline)-1):
            clr = 1 - (self.spline_velocity[i]-np.min(self.spline_velocity))/(np.max(self.spline_velocity) - np.min(self.spline_velocity))
            message.color.a=1.0
            message.color.r=clr
            message.color.b=clr
            message.color.g=clr

            message.id=i
            message.header.stamp = self.get_clock().now().to_msg()

            point1 = Point()
            point1.x = float(self.x_spline[i])
            point1.y = float(self.y_spline[i])
            point1.z = 0.0
            message.points.append(point1)

            point2 = Point()
            point2.x = float(self.x_spline[i+1])
            point2.y = float(self.y_spline[i+1])
            point2.z = 0.0
            message.points.append(point2)
            self.spline_publisher.publish(message)

        return message
    
    def visualize_mpc_path(self):

        message = Marker()
        message.header.frame_id="map"
        message.type= Marker.LINE_STRIP
        message.action = Marker.ADD
        message.pose.position.x= 0.0
        message.pose.position.y= 0.0
        message.pose.position.z=0.0
        message.pose.orientation.x=0.0
        message.pose.orientation.y=0.0
        message.pose.orientation.z=0.0
        message.pose.orientation.w=1.0
        message.scale.x=0.05
        message.color.r=1.0
        message.color.b=0.0
        message.color.g=0.0
        message.lifetime.nanosec=int(1e8)


        for i in range(len(self.ox)-1):
            message.color.a=1.0

            message.id=i
            message.header.stamp = self.get_clock().now().to_msg()

            point1 = Point()
            point1.x = float(self.ox[i])
            point1.y = float(self.oy[i])
            point1.z = 0.0
            message.points.append(point1)

            point2 = Point()
            point2.x = float(self.ox[i+1])
            point2.y = float(self.oy[i+1])
            point2.z = 0.0
            message.points.append(point2)
            self.spline_publisher.publish(message)
        return message
    
    def publish_rviz_data(self):
        self.goal_points_publisher.publish(self.pp_points_data)
        self.spline_publisher.publish(self.pp_spline_data)
        self.mpc_path_publisher.publish(self.mpc_spline_data)

    ########################### VISUALIZATION ############################
    
def global_2_local(quaternion, pt_w, T_c_w):
    # Transform goal point to vehicle frame of reference
    rot = (R.as_matrix(R.from_quat(quaternion)))
    pt_c = (np.array(pt_w) - np.array(T_c_w))@rot
    """ 
    # Alternate Method 
    H_global2car = np.zeros([4, 4]) #rigid body transformation from  the global frame of referce to the car
    H_global2car[3, 3] = 1
    current_rotation_matrix = R.from_quat(np.array([current_quat.x,current_quat.y,current_quat.z,current_quat.w])).as_matrix()
    H_global2car[0:3, 0:3] = np.array(current_rotation_matrix)
    H_global2car[0:3, 3] = np.array([current_position.x, current_position.y, current_position.z])

    # Calculate point
    goal_point_global = np.append(pt_w, 1).reshape(4, 1)
    pt_c = np.linalg.inv(H_global2car) @ goal_point_global
    """
    return pt_c

def load_points(file, scaler=10):
    # Open csv and read the waypoint data
    with open(file, 'r') as f:
        lines = (line for line in f if not line.startswith('#'))
        data = np.loadtxt(lines, delimiter=',', dtype=float)
    points = data / scaler

    return points
    
def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
