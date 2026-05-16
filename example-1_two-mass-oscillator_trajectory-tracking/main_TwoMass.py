import sys
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt


""" Define paths  """
# Path to FreeDyn dll
pathFDdll = 'C:\\VRoboCoop\\Programme\\FreeDyn\\freedyn_v1.0.6_preview\\freedyn.dll'

# Path to FreeDyn API """
pathFDApi = '..\\..\\freedyn\\bindings\\python'
sys.path.insert(0, pathFDApi)

# Path to core_opt_toolbox
bib_path = '..\\core_opt_toolbox'
sys.path.insert(0, bib_path)

# Define path and name of *.fds - without file typ!
path_fds = Path(__file__).resolve().parent
name_fds = 'OptCtrl_TwoMass'
    

"""  Choose OCP Problem """
# Final time tF is fixed: Optimal control problem with/without final constraints Phi
from class_OCP_MBS import Optimization

# Final time tF is free: Time-optimal control problem with/without final constraints Phi
# from class_OCP_MBS import Optimization     


""" Define the names of ctrl splines """
# derivative of sum of external forces w.r.t. parameter given as string
nameCtrlSpline = ["uDach"]


""" Define the names of the parameters for dfdu """
# derivative of sum of external forces w.r.t. parameter given as string
nameParFdu = ["fdu"]


""" Define controls """
numControls = 1          # number of controls
numGridNodes = 22        # number of grid nodes per control
uDachInit = np.zeros(numGridNodes*numControls)


""" Define final state of the MBS system """
tF = 10             # final time
xF = np.array([])   # final constraints, if no constraints are used, set = np.array([])


""" Define initial values for optimization variables z0"""
z0 = uDachInit      
numOptVar = len(z0)

optim = Optimization(numOptVar, numControls, numGridNodes,
                     tF, xF,
                     path_fds, name_fds,
                     nameCtrlSpline, nameParFdu,
                     pathFDdll)


options = {'disp': True, 'iprint': 2, 'ftol': 1e-8, 'eps':1e-8, 'maxiter': 150}

res = sp.optimize.minimize(fun         = optim.objective,                    # cost function
                           x0          = z0,                                 # initial values
                           method      = 'SLSQP',                            # optimization method
                           jac         = optim.get_grad_J,                   # gradient of cost function
                           options     = options                             # optimization options
                           )

# -----------------------------------------------------------------------------
optim.update_vars_if_changed(res.x)
optim.write_ctrl_dataSPL()

dyn_t = np.zeros(optim.dyn_numTimeSteps)
tau = np.zeros(optim.dyn_numTimeSteps)
uInit = np.zeros((numControls, optim.dyn_numTimeSteps))
u = np.zeros((numControls, optim.dyn_numTimeSteps))
y = np.zeros(optim.dyn_numTimeSteps)
ybar = np.zeros(optim.dyn_numTimeSteps)

for i in range(optim.dyn_numTimeSteps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   dyn_t[i] = optim.fd_model.t
   tau[i] = dyn_t[i]/optim.tF
   
   dyn_q = optim.fd_model.Q[:, 0]
   y[i] = dyn_q[7] - dyn_q[0]
   ybar[i] = optim.get_target_path(dyn_t[i])
   
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])


# -----------------------------------------------------------------------------
# plots
matplotlib.rcParams.update({'font.size': 15})
f = plt.figure(figsize=(10,5))

# plot relative motion
ax2 = f.add_subplot(1, 2, 1)
ax2.plot(dyn_t, y, linewidth = 1)
ax2.plot(dyn_t, ybar, linewidth = 1)
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

# -----------------------------------------------------------------------------

optim.__del__()
