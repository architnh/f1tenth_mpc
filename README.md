# f1tenth: Model Predictive Control

<img src=data/mpc_sim.gif height="300" width="300" > <p></p>
<!-- ## I. Learning Goals
- Convex Optimization
- Linearization and Discretization
- Optimal Control -->

## II. Formulating the MPC problem

The goal of the MPC is to generate valid input controls for $T$ steps ahead that controls the vehicle and follow the reference trajectory as close as possible.

### a. State and Dynamics Model

We use a kinematic model of the vehicle. The state space is $z=[x, y, \theta, v]$. And the input vector is $u=[a, \delta]$ where $a$ is acceleration and $\delta$ is steering angle. The kinematic model ODEs are:

$$
\dot{x}=v\cos(\theta)
$$

$$
\dot{y}=v\sin(\theta)
$$

$$
\dot{v}=a
$$

$$
\dot{\theta}=\frac{v\tan(\delta)}{L}
$$

Where $L$ is the wheelbase of the vehicle. In summary, we can write the continuous ODEs as:

$$
f(z, u)=A'z+B'u
$$

Here $A'$ and $B'$ are the original continuous system matrices. Finally, we discretize and linearize the system to get new system matrices $A$, $B$ and $C$.

### b. Objective Function

First we formulate the objective function of the optimization. We want to minimize three objectives:
1. Deviation of the vehicle from the reference trajectory. Final state deviation weighted by Qf, other state deviations weighted by Q.
2. Influence of the control inputs. Weighted by R.
3. Difference between one control input and the next control input. Weighted by Rd.

$$
\text{minimize}~~ Q_{f}\left(z_{T, r e f}-z_{T}\right)^{2}+Q \sum_{t=0}^{T-1}\left(z_{t, r e f}-z_{t}\right)^{2}+R \sum_{t=0}^{T} u_{t}^{2}+R_{d} \sum_{t=0}^{T-1}\left(u_{t+1}-u_{t}\right)^{2}
$$

### c. Constraints

We then formulate the constraints for the optimization problem. The constraints inclue:
1. Future vehicle states that follow the linearized vehicle dynamic model.
   $$z_{t+1}=Az_t+Bu_t+C$$
2. Initial state in the plan for the current horizon that matches the current vehicle state.
   $$z_{0}=z_{\text{curr}}$$
3. Inputs generated that are within vehicle limits.
   $$a_{\text{min}} \leq a \leq a_{\text{max}}$$
   $$\delta_{\text{min}} \leq \delta \leq \delta_{\text{max}}$$

## III. Linearization and Discretization

In order to formulate the problem into a Quadratic Programming (QP), we need to first discretize the dynamical system, and also linearize it around some point.

### a. Discretization

Now we use Forward Euler discretization since it's the easiest. Other methods like RK4/6 should also work. We discretize with sampling time $dt$, which we can pick as a parameter to tune. Thus, we can express the system equation as:

$$z_{t+1} = z_t + f(z_t, u_t)dt$$

### b. Linearization
We'll use first order Taylor expansion of the two variable function around some $\bar{z}$ and $\bar{u}$:

$$
z_{t+1}=z_t + f(z_t, u_t)dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + f'_z(\bar{z_t}, \bar{u_t})(z_t - \bar{z_t}) + f'_u(\bar{z_t}, \bar{u_t})(u_t - \bar{u_t}))dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + A'(z_t - \bar{z_t}) + B'(u_t - \bar{u_t}))dt
$$

$$
z_{t+1}=z_t + (f(\bar{z_t}, \bar{u_t}) + A'z_t - A'\bar{z_t} + B'u_t - B'\bar{u_t})dt
$$

$$
z_{t+1}=(I+dtA')z_t + dtB'u_t + (f(\bar{z_t}, \bar{u_t})- A'\bar{z_t} - B'\bar{u_t})dt
$$

$$
z_{t+1} = Az_t + Bu_t + C
$$

You can then derive what are matrices $A$, $B$, and $C$.

## IV. Reference Trajectory

We create a reference trajectory by hand picking points, and using scipy to generate a polynomial trajectory that fits these points. The curvature information can be found using the first derivative of the trajectory at the waypoints. Thus velocity at each point can be obtained. 

## V. Setting up the Optimization

We use CVXPY to set up the optimization problem with the OSQP solver. 

## VI. Visualization
Working Model Predictive Control on the f1tenth simulator.

