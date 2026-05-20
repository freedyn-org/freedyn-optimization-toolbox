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
# Path to core_opt_toolbox
path_optToolbox = str(path_main.parent.parent / 'core_opt_toolbox')
sys.path.insert(0, path_optToolbox)
#
# -----------------------------------------------------------------------------
#
""" FDS File """
# Define path and name of *.fds - without file typ!
path_fds = path_main
name_fds = 'OptCtrl_NonlinearSpringPendulum'

# Define FreeDyn data object spline of the controls
nameCtrlSpline = ["uDach_x", "uDach_y", "uDach_z"]

# Define FreeDyn parameter for fdu
nameParFdu = ["fdu_x","fdu_y","fdu_z"]
#
# -----------------------------------------------------------------------------
#
""" Define controls """
numControls = 3     # number of controls
numGridNodes = 10   # number of grid nodes per control

uDachInit = np.zeros(numGridNodes*numControls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
tF = 5                            # final time
xF = np.array([2,-10,-4,0,0,0])   # final constraints,
                                  # if no constraints are used, set = np.array([])
#
# -----------------------------------------------------------------------------
#
""" Define initial values for optimization variables z0"""
z0 = uDachInit      
numOptVar = len(z0)
#
# -----------------------------------------------------------------------------
#
"""  Choose OCP Problem """
# Optimal control problem with/without final constraints Phi
tF_free = 0   # is final time tF free? no ... 0 || yes ... 1

if tF_free:
    from class_TOCP_MBS import Optimization
else:
    from class_OCP_MBS import Optimization

optim = Optimization(numOptVar, numControls, numGridNodes,
                     tF, xF,
                     path_fds, name_fds,
                     nameCtrlSpline, nameParFdu,
                     path_FDdll)
#
# -----------------------------------------------------------------------------
#
"""  Set up of the optimization-toolbox """
# Add or comment out – according to the optimization problem
res = sp.optimize.minimize(fun         = optim.objective,                # cost function
                           x0          = z0,                             # initial values
                           method      = 'SLSQP',                        # optimization method
                           jac         = optim.get_grad_J,               # gradient of cost function
                           # bounds      = sp.optimize.Bounds(lb, ub),     # lower and upper bounds
                           constraints = {'type':'eq', 
                                          'fun':optim.ceq_tF, 
                                          'jac':optim.get_grad_Phi},     # non-linear constraints
                           options     = {'disp': True, 
                                          'iprint': 2, 
                                          'ftol': 1e-8, 
                                          'eps':1e-8, 
                                          'maxiter': 50}                 # optimization options
                           )
#
# -----------------------------------------------------------------------------
#
""" Update optimization variables in class and rerun simulation """
optim.update_vars_if_changed(res.x)
optim.change_tF_in_fds(optim.fds_path_name_optimized)
optim.write_ctrl_dataSPL()
#
# -----------------------------------------------------------------------------
#
""" Get data for plots """
t = np.zeros(optim.numTimeSteps)
tau = np.zeros(optim.numTimeSteps)
uInit = np.zeros((numControls, optim.numTimeSteps))
u = np.zeros((numControls, optim.numTimeSteps))
q = np.zeros((optim.nDof, optim.numTimeSteps))
qD = np.zeros((optim.nDof, optim.numTimeSteps))
spring_l = np.zeros(optim.numTimeSteps)

for i in range(optim.numTimeSteps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   optim.fd_model.update_state_at_index(i)
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   
   q[:,i] = optim.fd_model.Q[:, 0]
   qD[:,i] = optim.fd_model.Qd[:, 0]
   spring_l[i] = optim.fd_model.get_measure_value("l")
   
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])  
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 15})
f = plt.figure(figsize=(10,10))

# plot relative motion
ax1 = f.add_subplot(2,2, 1)
ax1.plot(t, q[0,:], c = 'blue', linewidth = 2)
ax1.plot(t, q[1,:], c = 'green', linewidth = 2)
ax1.plot(t, q[2,:], c = 'darkorange', linewidth = 2)
ax1.scatter(np.array([tF,tF,tF]), xF[0:3], c = 'r', marker = 'x')
ax1.set_xlabel('Time in s')
ax1.set_ylabel('Position in m')
ax1.grid()

# plot relative veloctiy
ax2 = f.add_subplot(2,2, 2)
ax2.plot(t, qD[0,:], c = 'blue', linewidth = 2)
ax2.plot(t, qD[1,:], c = 'green', linewidth = 2)
ax2.plot(t, qD[2,:], c = 'darkorange', linewidth = 2)
ax2.scatter(np.array([tF,tF,tF]), xF[3:6], c = 'r', marker = 'x')
ax2.set_xlabel('Time in s')
ax2.set_ylabel('Velocity in m/s')
ax2.grid()

# plot control
ax3 = f.add_subplot(2,2, 3)
ax3.plot(tau, u[0,:], c = 'blue', linewidth = 2)
ax3.plot(tau, u[1,:], c = 'green', linewidth = 2)
ax3.plot(tau, u[2,:], c = 'darkorange', linewidth = 2)
ax3.set_ylabel('control in Nm')
ax3.set_xlabel('normalized time')
ax3.grid()
ax3.set_xlim([0, 1])

# Plot Spring length
ax4 = f.add_subplot(2,2, 4)
ax4.plot(t, spring_l, c = 'blue', linewidth = 2)
ax4.set_ylabel('Spring length in m')
ax4.set_xlabel('Time in s')
ax4.grid()

plt.show()
#
# -----------------------------------------------------------------------------
#
optim.__del__()