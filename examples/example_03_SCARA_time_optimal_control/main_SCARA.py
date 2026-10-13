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
path_folder_optim_toolbox = str(path_main.parent.parent)
sys.path.insert(0, path_folder_optim_toolbox)
#
# -----------------------------------------------------------------------------
#
""" FDS - File """
fds_path = path_main                                    # Path of *.fds
fds_name = 'OptCtrl_SCARA'                              # Name of *.fds (without file typ)
fds_dForce_dparam_name = ["u1par","u2par"]              # FreeDyn param for df / dparam
fds_meas_name = None                                    # FreeDyn measures
fds_ctrl_name = ["u1Dach", "u2Dach"]                    # FreeDyn control splines
#
# -----------------------------------------------------------------------------
#
""" Define controls """
num_ctrls = 2     # number of controls
num_ctrl_grid_nodes = 50   # number of grid nodes per control

uDachInit = np.zeros(num_ctrl_grid_nodes*num_ctrls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
final_time_init = 3                                # final time
final_state = np.array([1.0, 1.0, 0, 0])           # final constraints | # if no, set: None
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
zInit = np.append(final_time_init, uDachInit)
#
# -----------------------------------------------------------------------------
#
""" Define bounds """
uLimit = np.array([4,2])        # control limit
lb = np.array(0.01)
ub = np.array(np.inf) 

for loop_Limit in range(0, num_ctrls):
    lb = np.append(lb, -uLimit[loop_Limit]*np.ones(num_ctrl_grid_nodes))
    ub = np.append(ub, uLimit[loop_Limit]*np.ones(num_ctrl_grid_nodes))
#
# -----------------------------------------------------------------------------
#
"""  Choose Optimization with/without final constraints Phi """
from optimization_toolbox import OptimCtrl, TimeOptimCtrl, ParamIdent
# Use "OptimCtrl" for Optimal Control Problems with fixed final time
# Use "TimeOptimCtrl" for Optimal Control Problems with free final time
# Use "ParamIdent" for Parameter-Identification with fixed final time
#
Opt_TB = TimeOptimCtrl(fds_path, fds_name,
                       fds_dForce_dparam_name, fds_meas_name, fds_ctrl_name,
                       num_ctrls, num_ctrl_grid_nodes, num_FD_pars,
                       final_time_init, final_state,
                       path_FDdll)
#
# -----------------------------------------------------------------------------
#
"""  Set up of the optimization-toolbox """
# Add or comment out – according to the optimization problem
res = sp.optimize.minimize(fun         = Opt_TB.cost_fct_J,                    # cost function
                           x0          = zInit,                                # initial values
                           method      = 'SLSQP',                              # optimization method
                           jac         = Opt_TB.grad_cost_fct_J,               # gradient of cost function
                           bounds      = sp.optimize.Bounds(lb, ub),           # lower and upper bounds
                           constraints = {'type':'eq',                         # non-linear constraints
                                          'fun':Opt_TB.final_constr_eq_Phi, 
                                          'jac':Opt_TB.grad_final_constr_eq_Phi},
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 500}                      # optimization options
                           ) 
#
# -----------------------------------------------------------------------------
#
""" Update optimization variables in class and rerun simulation """
Opt_TB.new_opt_vars(res.x)
# Opt_TB.FreeDyn.write_ctrl_dataSPL()
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(Opt_TB.FreeDyn.num_time_steps)                    # physical time t
tau = np.zeros(Opt_TB.FreeDyn.num_time_steps)                  # normalized time scale [0;1]
uInit = np.zeros((num_ctrls, Opt_TB.FreeDyn.num_time_steps))   # initial control
u = np.zeros((num_ctrls, Opt_TB.FreeDyn.num_time_steps))       # optimal control
q = np.zeros((Opt_TB.FreeDyn.nDof, Opt_TB.FreeDyn.num_time_steps))      # gen. red. coordinates
qD = np.zeros((Opt_TB.FreeDyn.nDof, Opt_TB.FreeDyn.num_time_steps))     # gen. red. velocities
    
for i in range(Opt_TB.FreeDyn.num_time_steps-1, -1, -1): 
   Opt_TB.FreeDyn.fetch_and_update_states_at_index(i)
   t[i] = Opt_TB.FreeDyn.API.t
   tau[i] = t[i]/Opt_TB.OptimTask.final_time
   uInit[:,i] = Opt_TB.Ctrl.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = Opt_TB.Ctrl.get_u(tau[i])
   q[:,i] = Opt_TB.FreeDyn.API.Q[:, 0]
   qD[:,i] = Opt_TB.FreeDyn.API.Qd[:, 0]

x_TCP = q[14,:]
y_TCP = q[15,:]
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 12})
f = plt.figure(figsize=(12,4))

# plot position of TCP
ax1 = f.add_subplot(1, 2, 1)
ax1.plot(x_TCP, y_TCP, linewidth = 1, label = "x_TCP")
ax1.scatter(x_TCP[0], y_TCP[1], c = 'g', marker  = 'x', label = "x_TCP(t_0)")
ax1.scatter(final_state[0], final_state[1], c = 'r', marker = 'x', label = "x_TCP(t_f)")
ax1.set_xlabel('x in m')
ax1.set_ylabel('y in m')
ax1.legend(loc='upper right',ncols = 1)
ax1.grid()

# plot control
ax2 = f.add_subplot(1, 2, 2)
ax2.plot(tau, u[0,:].T, linewidth = 2, label = "u1")
ax2.plot(tau, u[1,:].T, linewidth = 2, label = "u2")
ax2.scatter(np.linspace(0, 1, num_ctrl_grid_nodes),  uLimit[0]*np.ones(num_ctrl_grid_nodes), c = 'b', marker = 'x')
ax2.scatter(np.linspace(0, 1, num_ctrl_grid_nodes), -uLimit[0]*np.ones(num_ctrl_grid_nodes), c = 'b', marker = 'x')
ax2.scatter(np.linspace(0, 1, num_ctrl_grid_nodes),  uLimit[1]*np.ones(num_ctrl_grid_nodes), c = 'orange', marker = 'x')
ax2.scatter(np.linspace(0, 1, num_ctrl_grid_nodes), -uLimit[1]*np.ones(num_ctrl_grid_nodes), c = 'orange', marker = 'x')
ax2.set_ylabel('control in Nm')
ax2.set_xlabel('normalized time')
ax2.grid()
ax2.legend(loc='upper right',ncols = 1)
ax2.set_xlim([0, 1])

plt.show()
#
# -----------------------------------------------------------------------------
#
Opt_TB.FreeDyn.delete_model()