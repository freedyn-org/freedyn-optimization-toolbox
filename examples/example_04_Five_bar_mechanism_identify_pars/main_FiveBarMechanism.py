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
fds_path = path_main                                       # Path of *.fds
fds_name = 'five_bar_planar_mechanism'                     # Name of *.fds (without file typ)
fds_dForce_dparam_name = ["l0_spring1", "l0_spring2"]      # FreeDyn param for df / dparam
fds_meas_name = ["dispX_P2", "dispY_P2"]                   # FreeDyn measures
fds_ctrl_name = None,                                      # FreeDyn control splines
#
# -----------------------------------------------------------------------------
#
""" Define controls """
num_ctrls = 0     # number of controls
num_ctrl_grid_nodes = 0   # number of grid nodes per control

uDachInit = np.zeros(num_ctrl_grid_nodes*num_ctrls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
final_time = 5             # final time
final_state = np.array([])   # final constraints,
                    # if no xF are used, set: xF = np.array([])
#
# -----------------------------------------------------------------------------
#
""" Define FD parameters - if required"""
l_spring_1 = np.sqrt(5)
l_spring_2 = np.sqrt(4.25)
FD_pars = np.array([l_spring_1, l_spring_2])
num_FD_pars = len(FD_pars)
#
# -----------------------------------------------------------------------------
#
""" Define initial values for optimization variables zInit"""
zInit = FD_pars
#
# -----------------------------------------------------------------------------
#
"""  Choose Optimization with/without final constraints Phi """
from optimization_toolbox import OptimCtrl, TimeOptimCtrl, ParamIdent
# Use "OptimCtrl" for Optimal Control Problems with fixed final time
# Use "TimeOptimCtrl" for Optimal Control Problems with free final time
# Use "ParamIdent" for Parameter-Identification with fixed final time
#
Opt_TB = ParamIdent(fds_path, fds_name,
                    fds_dForce_dparam_name, fds_meas_name, fds_ctrl_name,
                    num_ctrls, num_ctrl_grid_nodes, num_FD_pars,
                    final_time, final_state,
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
                           # bounds      = sp.optimize.Bounds(lb, ub),           # lower and upper bounds
                           # constraints = {'type':'eq',                         # non-linear constraints
                           #                'fun':Opt_TB.final_constr_eq_Phi, 
                           #                'jac':Opt_TB.grad_final_constr_eq_Phi},
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 50}                       # optimization options
                           )
#
# -----------------------------------------------------------------------------
#
# """ Update optimization variables in class and rerun simulation """
Opt_TB.new_opt_vars(res.x)

for i in range(0,num_FD_pars):
    print(f'{fds_dForce_dparam_name[i]} = {res.x[i]}')
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(Opt_TB.FreeDyn.num_time_steps)                    # physical time t
tau = np.zeros(Opt_TB.FreeDyn.num_time_steps)                  # normalized time scale [0;1]
rx_P2 =  np.zeros(Opt_TB.FreeDyn.num_time_steps)           # 
ry_P2 =  np.zeros(Opt_TB.FreeDyn.num_time_steps)            # 
vx_P2 =  np.zeros(Opt_TB.FreeDyn.num_time_steps)            # 
vy_P2 =  np.zeros(Opt_TB.FreeDyn.num_time_steps)            # 

for i in range(Opt_TB.FreeDyn.num_time_steps-1, -1, -1): 
   Opt_TB.FreeDyn.fetch_and_update_states_at_index(i)
   t[i] = Opt_TB.FreeDyn.API.t
   tau[i] = t[i]/Opt_TB.OptimTask.final_time
   q = Opt_TB.FreeDyn.API.Q[:, 0]
   
   rx_P2[i] = Opt_TB.FreeDyn.API.get_measure_value("dispX_P2")
   ry_P2[i] = Opt_TB.FreeDyn.API.get_measure_value("dispY_P2")
   vx_P2[i] = Opt_TB.FreeDyn.API.get_measure_value("veloX_P2")
   vy_P2[i] = Opt_TB.FreeDyn.API.get_measure_value("veloY_P2")
   
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
Opt_TB.FreeDyn.delete_model()