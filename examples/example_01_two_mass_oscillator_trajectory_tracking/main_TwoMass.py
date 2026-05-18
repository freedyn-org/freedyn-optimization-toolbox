import sys
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
#
# -----------------------------------------------------------------------------
""""
.\FreeDyn 
  ├── Releases\
  │   └── freedyn-106   !!! MUST NOT USE POINTS HERE !!!
  │       ├── bin\
  │       │   └── FreeDyn-win-x64_MD   # MD variant: Freedyn_GUI.exe, freedyn.dll, dependencies
  │       │       └── freedyn.dll
  │       └── bindings
  │           └── python               # Python API (source code)
  └── freedyn-optimization-toolbox
      ├── core_opt_toolbox             # global methods for optimization
      └── examples
          ├── example_01  <- current working directory
          ├── example_02
          └── example_03
"""
# -----------------------------------------------------------------------------
#
""" Define paths  """
# Path of main
path_main = Path(__file__).resolve().parent

# Path to FreeDyn dll
pathFDdll = path_main.parent.parent.parent / 'Releases\\freedyn-106\\bin\\FreeDyn-win-x64_MD\\freedyn.dll'

# Path to FreeDyn API 
path_FDApi = path_main.parent.parent.parent / 'Releases\\freedyn-106\\bindings\\python'
sys.path.insert(0, str(path_FDApi))

# Path to core_opt_toolbox
bib_path = path_main.parent.parent / 'core_opt_toolbox'
sys.path.insert(0, str(bib_path))

# Define path and name of *.fds - without file typ!
path_fds = path_main
name_fds = 'OptCtrl_TwoMass'
#
# -----------------------------------------------------------------------------
#
""" Define the names of ctrl splines """
# derivative of sum of external forces w.r.t. parameter given as string
nameCtrlSpline = ["uDach"]
#
# -----------------------------------------------------------------------------
#
""" Define the names of the parameters for dfdu """
# derivative of sum of external forces w.r.t. parameter given as string
nameParFdu = ["fdu"]
#
# -----------------------------------------------------------------------------
#
""" Define controls """
numControls = 1     # number of controls
numGridNodes = 22   # number of grid nodes per control

uDachInit = np.zeros(numGridNodes*numControls)
#
# -----------------------------------------------------------------------------
#
""" Define final state of the MBS system """
tF = 10             # final time
xF = np.array([])   # final constraints,
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
                     pathFDdll)
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
                           # constraints = {'type':'eq', 
                           #                'fun':optim.ceq_tF, 
                           #                'jac':optim.get_grad_Phi},     # non-linear constraints
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
y = np.zeros(optim.numTimeSteps)
ybar = np.zeros(optim.numTimeSteps)

for i in range(optim.numTimeSteps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   
   q = optim.fd_model.Q[:, 0]
   y[i] = q[7] - q[0]
   ybar[i] = optim.get_target_path(t[i])
   
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])
#
# -----------------------------------------------------------------------------
#
""" Plots """
matplotlib.rcParams.update({'font.size': 15})
f = plt.figure(figsize=(10,5))

# plot relative motion
ax2 = f.add_subplot(1, 2, 1)
ax2.plot(t, y, linewidth = 1)
ax2.plot(t, ybar, linewidth = 1)
ax2.set_xlabel('normalized time')
ax2.set_ylabel('y in m')
ax2.grid()

# plot control
ax1 = f.add_subplot(1, 2, 2)
ax1.plot(tau, u.T, linewidth = 2)
ax1.set_ylabel('control in N')
ax1.set_xlabel('normalized time')
ax1.grid()
ax1.set_xlim([0, 1])

plt.show()
#
# -----------------------------------------------------------------------------
#
optim.__del__()