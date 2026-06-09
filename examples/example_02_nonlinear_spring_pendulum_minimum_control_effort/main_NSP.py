import sys
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
#
# -----------------------------------------------------------------------------
#
# Path of main file
path_main = Path(__file__).resolve().parent
#
# -----------------------------------------------------------------------------
#
""" FreeDyn dll and Python bindings paths  """
# Path to FreeDyn dll
# Use None: if pip install freedyn
# Define path when Python bindings are download from GitHub without installing
path_FDdll = None

# Path to FreeDyn API 
# Use None: if pip install freedyn
# Define path when Python bindings are download from GitHub without installing
path_FDApi = None

if path_FDApi is not None:
    sys.path.insert(0, path_FDApi)
#
# -----------------------------------------------------------------------------
#
""" Freedyn Optimization Toolbox """
# Path to optimization_toolbox
path_optToolbox = str(path_main.parent.parent / 'optimization_toolbox')
sys.path.insert(0, path_optToolbox)
#
# -----------------------------------------------------------------------------
#
""" FDS File """
# Define path and name of *.fds - without file typ!
path_fds = path_main
name_fds = 'OptCtrl_NonlinearSpringPendulum'

# Define FreeDyn data object spline of the controls
name_ctrlSPL = ["uDach_x", "uDach_y", "uDach_z"]

# Define FreeDyn parameter for fdu
name_fDu_par = ["fdu_x","fdu_y","fdu_z"]
#
# -----------------------------------------------------------------------------
#
""" Define controls """
num_ctrls = 3     # number of controls
num_ctrl_gridNodes = 10   # number of grid nodes per control

uDachInit = np.zeros(num_ctrl_gridNodes*num_ctrls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
tF = 5                            # final time
xF = np.array([2,-10,-4,0,0,0])   # final constraints,
                                  # if no xF are used, set: xF = np.array([])
#
# -----------------------------------------------------------------------------
#
""" Define initial values for optimization variables zInit"""
zInit = uDachInit      
num_optVars = len(zInit)
#
# -----------------------------------------------------------------------------
#
"""  Choose Optimal Control Problem (OCP) with/without final constraints Phi """
# Commenting in and out – according to the optimization problem

from class_OCP_FDOP import Optimization   # OCP with fixed final time 
#  from class_TOCP_FDOP import Optimization  # OCP with free final time 

optim = Optimization(num_optVars, num_ctrls, num_ctrl_gridNodes,
                     tF, xF,
                     path_fds, name_fds,
                     name_ctrlSPL, name_fDu_par,
                     path_FDdll)
#
# -----------------------------------------------------------------------------
#
"""  Set up of the optimization-toolbox """
# Add or comment out – according to the optimization problem
res = sp.optimize.minimize(fun         = optim.costFct_J,                # cost function
                           x0          = zInit,                             # initial values
                           method      = 'SLSQP',                        # optimization method
                           jac         = optim.grad_costFct_J,               # gradient of cost function
                           # bounds      = sp.optimize.Bounds(lb, ub),     # lower and upper bounds
                           constraints = {'type':'eq', 
                                          'fun':optim.finalConstr_Phi, 
                                          'jac':optim.grad_finalConstr_Phi},     # non-linear constraints
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 100}                 # optimization options
                           )
#
# -----------------------------------------------------------------------------
#
""" Update optimization variables in class and rerun simulation """
optim.update_vars_if_changed(res.x)
# optim.write_ctrl_dataSPL()
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(optim.num_time_steps)                    # physical time t
tau = np.zeros(optim.num_time_steps)                  # normalized time scale [0;1]
uInit = np.zeros((num_ctrls, optim.num_time_steps))   # initial control
u = np.zeros((num_ctrls, optim.num_time_steps))       # optimal control
q = np.zeros((optim.nDof, optim.num_time_steps))      # gen. red. coordinates
qD = np.zeros((optim.nDof, optim.num_time_steps))     # gen. red. velocities
spring_l = np.zeros(optim.num_time_steps)             # length of the spring

for i in range(optim.num_time_steps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   optim.fd_model.update_state_at_index(i)
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])  
   q[:,i] = optim.fd_model.Q[:, 0]
   qD[:,i] = optim.fd_model.Qd[:, 0]
   spring_l[i] = optim.fd_model.get_measure_value("l") 
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 12})
f = plt.figure(figsize=(12,8))

# plot relative motion
ax1 = f.add_subplot(2,2,1)
ax1.plot(t, q[0,:], c = 'blue', linewidth = 2, label = "x")
ax1.plot(t, q[1,:], c = 'green', linewidth = 2, label = "y")
ax1.plot(t, q[2,:], c = 'darkorange', linewidth = 2, label = "z")
ax1.scatter(np.array([tF,tF,tF]), xF[0:3], c = 'r', marker = 'x', label = "xf")
ax1.set_xlabel('Time in s')
ax1.set_ylabel('Position in m')
ax1.legend(loc='upper left',ncols = 2)
ax1.grid()

# plot relative veloctiy
ax2 = f.add_subplot(2,2,2)
ax2.plot(t, qD[0,:], c = 'blue', linewidth = 2, label = "vx")
ax2.plot(t, qD[1,:], c = 'green', linewidth = 2, label = "vy")
ax2.plot(t, qD[2,:], c = 'darkorange', linewidth = 2, label = "vz")
ax2.scatter(np.array([tF,tF,tF]), xF[3:6], c = 'r', marker = 'x', label = "vf")
ax2.set_xlabel('Time in s')
ax2.set_ylabel('Velocity in m/s')
ax2.legend(loc='lower left',ncols = 4)
ax2.grid()

# plot control
ax3 = f.add_subplot(2,2,3)
ax3.plot(tau, u[0,:], c = 'blue', linewidth = 2, label = "ux")
ax3.plot(tau, u[1,:], c = 'green', linewidth = 2, label = "uy")
ax3.plot(tau, u[2,:], c = 'darkorange', linewidth = 2, label = "uz")
ax3.set_ylabel('control in Nm')
ax3.set_xlabel('normalized time')
ax3.grid()
ax3.legend(loc='upper center',ncols = 3)
ax3.set_xlim([0, 1])

# Plot Spring length
ax4 = f.add_subplot(2,2,4)
ax4.plot(t, spring_l, c = 'blue', linewidth = 2)
ax4.set_ylabel('Spring length in m')
ax4.set_xlabel('Time in s')
ax4.grid()

plt.show()
#
# -----------------------------------------------------------------------------
#
optim.__del__()