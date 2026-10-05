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

# Define FreeDyn measures
name_fDmeas = []

# Define FreeDyn parameter for fdu
name_dForce_dparam = ["fdu"]
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
""" Define FD parameters - if required"""
FD_pars = np.array([])
num_FD_pars = len(FD_pars)
#
# -----------------------------------------------------------------------------
#
""" Define initial values for optimization variables zInit"""
zInit = uDachInit      
num_optVars = len(zInit)
#
# -----------------------------------------------------------------------------
#
"""  Choose Optimization with/without final constraints Phi """
from optimization_toolbox import Toolbox
# Use "OCP" for Optimal Control Problems with fixed final time
# Use "TOCP" for Optimal Control Problems with free final time
# Use "Parameter" for Parameter-Identification with fixed final time
#
opt_TB = Toolbox("OCP", 
                 num_optVars, num_ctrls, num_ctrl_gridNodes, num_FD_pars,
                 tF, xF,
                 path_fds, name_fds,
                 name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                 path_FDdll)
#
# -----------------------------------------------------------------------------
#
"""  Set up of the optimization-toolbox """
# Add or comment out – according to the optimization problem
res = sp.optimize.minimize(fun         = opt_TB.costFct_J,                # cost function
                           x0          = zInit,                             # initial values
                           method      = 'SLSQP',                        # optimization method
                           jac         = opt_TB.grad_costFct_J,               # gradient of cost function
                           # bounds      = sp.optimize.Bounds(lb, ub),     # lower and upper bounds
                           # constraints = {'type':'eq', 
                           #                'fun':opt_TB.finalConstr_Phi, 
                           #                'jac':opt_TB.grad_finalConstr_Phi},     # non-linear constraints
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 5}                # optimization options
                           )
#
# -----------------------------------------------------------------------------
#
""" Update optimization variables in class and rerun simulation """
opt_TB.optim_vars.update_optim_vars.check_if_changed(opt_TB.opt_task, opt_TB.FreeDyn, opt_TB.ctrl, res.x)
# opt_TB.FreeDyn.write_ctrl_dataSPL()
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(opt_TB.FreeDyn.num_time_steps)                    # physical time t
tau = np.zeros(opt_TB.FreeDyn.num_time_steps)                  # normalized time scale [0;1]
uInit = np.zeros((num_ctrls, opt_TB.FreeDyn.num_time_steps))   # initial control
u = np.zeros((num_ctrls, opt_TB.FreeDyn.num_time_steps))       # optimal control
y = np.zeros(opt_TB.FreeDyn.num_time_steps)                    # x2 - x1
ybar = np.zeros(opt_TB.FreeDyn.num_time_steps)                 # target path

for i in range(opt_TB.FreeDyn.num_time_steps-1, -1, -1): 
   opt_TB.FreeDyn.API.fetch_states_at_index(i)
   t[i] = opt_TB.FreeDyn.API.t
   tau[i] = t[i]/opt_TB.opt_task.final_time
   uInit[:,i] = opt_TB.ctrl.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = opt_TB.ctrl.get_u(tau[i])
   q = opt_TB.FreeDyn.API.Q[:, 0]
   y[i] = q[7] - q[0]
   ybar[i] = opt_TB.opt_task.user_Fcts.get_target_path(t[i])  
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
opt_TB.FreeDyn.delete_model()