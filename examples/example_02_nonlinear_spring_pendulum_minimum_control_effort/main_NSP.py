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

# Define FreeDyn measures
name_fDmeas = []

# Define FreeDyn parameter for fdu
name_dForce_dparam = ["fdu_x","fdu_y","fdu_z"]
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
Opt_TB = Toolbox("OCP",
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
res = sp.optimize.minimize(fun         = Opt_TB.cost_fct_J,                    # cost function
                           x0          = zInit,                                # initial values
                           method      = 'SLSQP',                              # optimization method
                           jac         = Opt_TB.grad_cost_fct_J,               # gradient of cost function
                           # bounds      = sp.optimize.Bounds(lb, ub),           # lower and upper bounds
                           constraints = {'type':'eq',                         # non-linear constraints
                                          'fun':Opt_TB.final_constr_eq_Phi, 
                                          'jac':Opt_TB.grad_final_constr_eq_Phi},
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 50}                       # optimization options
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
spring_l = np.zeros(Opt_TB.FreeDyn.num_time_steps)             # length of the spring

for i in range(Opt_TB.FreeDyn.num_time_steps-1, -1, -1): 
   Opt_TB.FreeDyn.fetch_and_update_states_at_index(i)
   t[i] = Opt_TB.FreeDyn.API.t
   tau[i] = t[i]/Opt_TB.OptimTask.final_time
   uInit[:,i] = Opt_TB.Ctrl.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = Opt_TB.Ctrl.get_u(tau[i])  
   q[:,i] = Opt_TB.FreeDyn.API.Q[:, 0]
   qD[:,i] = Opt_TB.FreeDyn.API.Qd[:, 0]
   spring_l[i] = Opt_TB.FreeDyn.API.get_measure_value("l") 
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
Opt_TB.FreeDyn.delete_model()