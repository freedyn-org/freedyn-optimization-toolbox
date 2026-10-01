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
name_fds = 'five_bar_planar_mechanism'

# Define FreeDyn data object spline of the controls
name_ctrlSPL = []

# Define FreeDyn parameter for fdu
# name_fDu_par = ["fdu"]

# Define FreeDyn parameter for Parameters
name_fDpar = ["springOneLengthNature","springTwoLengthNature"]


# Define FreeDyn measures
name_fDmeas = ["dispX_P2","dispY_P2"]
#
# -----------------------------------------------------------------------------
#
""" Define controls """
num_ctrls = 0     # number of controls
num_ctrl_gridNodes = 0   # number of grid nodes per control

uDachInit = np.zeros(num_ctrl_gridNodes*num_ctrls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
tF = 5             # final time
xF = np.array([])   # final constraints,
                    # if no xF are used, set: xF = np.array([])
#
# -----------------------------------------------------------------------------
#
""" Define initial values for optimization variables zInit"""
l_spring_1 = np.sqrt(5)
l_spring_2 = np.sqrt(4.25)
pars = np.array([l_spring_1, l_spring_2])

zInit = pars
num_optVars = len(zInit)
#
# -----------------------------------------------------------------------------
#
"""  Choose Optimal Control Problem (OCP) with/without final constraints Phi """
# Commenting in and out – according to the optimization problem

#  from class_OCP_FDOP import Optimization   # OCP with fixed final time 
#  from class_TOCP_FDOP import Optimization  # OCP with free final time 
from class_Identify_Pars import Optimization

optim = Optimization(num_optVars, num_ctrls, num_ctrl_gridNodes,
                     tF, xF,
                     path_fds, name_fds,
                     name_ctrlSPL, name_fDpar,
                     name_fDmeas,
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
                           # constraints = {'type':'eq', 
                           #                'fun':optim.finalConstr_Phi, 
                           #                'jac':optim.grad_finalConstr_Phi},     # non-linear constraints
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 50}                # optimization options
                           )
#
# -----------------------------------------------------------------------------
#
""" Update optimization variables in class and rerun simulation """
optim.update_vars_if_changed(res.x)

for i in range(0,num_optVars):
    print(f'{name_fDpar[i]} = {res.x[i]}')
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(optim.num_time_steps)                    # physical time t
tau = np.zeros(optim.num_time_steps)                  # normalized time scale [0;1]
rx_P2 =  np.zeros(optim.num_time_steps)           # 
ry_P2 =  np.zeros(optim.num_time_steps)            # 
vx_P2 =  np.zeros(optim.num_time_steps)            # 
vy_P2 =  np.zeros(optim.num_time_steps)            # 

for i in range(optim.num_time_steps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   optim.fd_model.update_state_at_index(i)   # necessary, if measures are used in get_Lagrangian()
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   q = optim.fd_model.Q[:, 0]
   
   rx_P2[i] = optim.fd_model.get_measure_value("dispX_P2")
   ry_P2[i] = optim.fd_model.get_measure_value("dispY_P2")
   vx_P2[i] = optim.fd_model.get_measure_value("veloX_P2")
   vy_P2[i] = optim.fd_model.get_measure_value("veloY_P2")
   
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 12})
f = plt.figure(figsize=(12,4))

# plot position Point 2
ax1 = f.add_subplot(1, 2, 1)
ax1.plot(t, rx_P2, linewidth = 1, label = "P2_x")
ax1.plot(t, ry_P2, linewidth = 1, label = "P2_y")
ax1.set_xlabel('Time in s')
ax1.set_ylabel('Position in m')
ax1.legend(loc='upper right',ncols = 2)
ax1.grid()

# plot velocity Point 2
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(t, vx_P2, linewidth = 1, label = "P2_vx")
ax2.plot(t, vy_P2, linewidth = 1, label = "P2_vy")
ax2.set_xlabel('Time in s')
ax2.set_ylabel('Velocity in m/s')
ax2.legend(loc='upper right',ncols = 2)
ax2.grid()

plt.show()
#
# -----------------------------------------------------------------------------
#
optim.__del__()