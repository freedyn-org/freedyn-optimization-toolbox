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
name_fds = 'OptCtrl_TwoMass'

# Define FreeDyn data object spline of the controls
name_ctrlSPL = ["uDach"]

# Define FreeDyn parameter for fdu
name_fDu_par = ["fdu"]
#
# -----------------------------------------------------------------------------
#
""" Define controls """
num_ctrls = 1     # number of controls
num_ctrl_gridNodes = 22   # number of grid nodes per control

uDachInit = np.zeros(num_ctrl_gridNodes*num_ctrls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
tF = 10             # final time
xF = np.array([])   # final constraints,
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
                           # constraints = {'type':'eq', 
                           #                'fun':optim.finalConstr_Phi, 
                           #                'jac':optim.grad_finalConstr_Phi},     # non-linear constraints
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 150}                # optimization options
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
y = np.zeros(optim.num_time_steps)                    # x2 - x1
ybar = np.zeros(optim.num_time_steps)                 # target path

for i in range(optim.num_time_steps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])
   q = optim.fd_model.Q[:, 0]
   y[i] = q[7] - q[0]
   ybar[i] = optim.get_target_path(t[i])  
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 12})
f = plt.figure(figsize=(12,4))

# plot relative motion
ax1 = f.add_subplot(1, 2, 1)
ax1.plot(t, y, linewidth = 1, label = "y")
ax1.plot(t, ybar, '--', linewidth = 1, label = "y_bar")
ax1.set_xlabel('normalized time')
ax1.set_ylabel('y in m')
ax1.legend(loc='upper right',ncols = 2)
ax1.grid()

# plot control
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(tau, u.T, linewidth = 2)
ax2.set_ylabel('control in N')
ax2.set_xlabel('normalized time')
ax2.grid()
ax2.set_xlim([0, 1])

plt.show()
#
# -----------------------------------------------------------------------------
#
optim.__del__()