import sys
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import time
from pyinstrument import Profiler


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
name_fds = 'OptCtrl_SCARA'


"""  Choose OCP Problem """
# Final time tF is fixed: Optimal control problem with/without final constraints Phi
# from class_OCP_MBS import Optimization

# Final time tF is free: Time-optimal control problem with/without final constraints Phi
from class_TOCP_MBS import Optimization     


""" Define the names of ctrl splines """
# derivative of sum of external forces w.r.t. parameter given as string
nameCtrlSpline = ["u1Dach", "u2Dach"]


""" Define the names of the parameters for dfdu """
# derivative of sum of external forces w.r.t. parameter given as string
nameParFdu = ["u1par","u2par"]


""" Define controls """
numControls = 2           # number of controls
numGridNodes = 50        # number of grid nodes per control
uDachInit = np.zeros(numGridNodes*numControls)


""" Define final state of the MBS system """
tF_init = 3             # final time
xF = np.array([1.0, 1.0, 0, 0])   # final constraints, if no constraints are used, set = np.array([])


""" Define initial values for optimization variables z0"""
z0 = np.append(tF_init, uDachInit)    
numOptVar = len(z0)


""" Define bounds """
uLimit = np.array([4,2])        # control limit
lb = np.array(0.01)
ub = np.array(np.inf) 

for loop_Limit in range(0, numControls):
    lb = np.append(lb, -uLimit[loop_Limit]*np.ones(numGridNodes))
    ub = np.append(ub, uLimit[loop_Limit]*np.ones(numGridNodes))
  
bounds = sp.optimize.Bounds(lb, ub)




optim = Optimization(numOptVar, numControls, numGridNodes, 
                     tF_init, xF,
                     path_fds, name_fds,
                     nameCtrlSpline, nameParFdu,
                     pathFDdll)


options = {'disp': True, 'iprint': 2, 'ftol': 1e-8, 'eps':1e-8, 'maxiter': 500}
constraints = {'type':'eq', 'fun':optim.ceq_tF, 'jac':optim.get_grad_Phi}

profiler = Profiler()
profiler.start()
countStart = time.perf_counter()
res = sp.optimize.minimize(fun         = optim.objective,                    # cost function
                           x0          = z0,                                 # initial values
                           method      = 'SLSQP',                            # optimization method
                           jac         = optim.get_grad_J,                   # gradient of cost function
                           bounds      = bounds,                             # lower and upper bounds
                           constraints = constraints,                        # non-linear constraints
                           options     = options                             # optimization options
                           )
countEnd = time.perf_counter()
profiler.stop()
timeComp = countEnd - countStart
print(f"Zeitdauer Optimierung: {timeComp} s")

profiler.open_in_browser("speedscope")  # Öffnet direkt in speedscope.app
    
# -----------------------------------------------------------------------------
optim.update_vars_if_changed(res.x)
optim.write_ctrl_dataSPL()

t = np.zeros(optim.numTimeSteps)
tau = np.zeros(optim.numTimeSteps)
uInit = np.zeros((numControls, optim.numTimeSteps))
u = np.zeros((numControls, optim.numTimeSteps))
q = np.zeros((optim.nDof, optim.numTimeSteps))
qD = np.zeros((optim.nDof, optim.numTimeSteps))
    
for i in range(optim.numTimeSteps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   t[i] = optim.fd_model.t
   tau[i] = t[i]/optim.tF
   
   uInit[:,i] = optim.get_u_for_GridNodes(tau[i], uDachInit)
   u[:,i] = optim.get_u(tau[i])
   q[:,i] = optim.fd_model.Q[:, 0]
   qD[:,i] = optim.fd_model.Qd[:, 0]


x_TCP = q[14,:]
y_TCP = q[15,:]
x_COG2 = q[7,:]
y_COG2 = q[8,:]
x_COG1 = q[0,:]
y_COG1 = q[1,:]

# -----------------------------------------------------------------------------
# plots
matplotlib.rcParams.update({'font.size': 15})
f = plt.figure(figsize=(12,5))

# plot position of TCP
ax2 = f.add_subplot(1, 2, 1)
ax2.plot(x_TCP, y_TCP, linewidth = 1)
ax2.plot(x_COG2, y_COG2, linewidth = 1)
ax2.plot(x_COG1, y_COG1, linewidth = 1)
ax2.scatter(xF[0], xF[1], c = 'r', marker = 'x')
ax2.set_xlabel('x in m')
ax2.set_ylabel('y in m')
ax2.grid()

# plot control
ax1 = f.add_subplot(1, 2, 2)
ax1.plot(tau, u.T, linewidth = 2)
ax1.scatter(np.linspace(0, 1, numGridNodes),  uLimit[0]*np.ones(numGridNodes), c = 'b', marker = 'x')
ax1.scatter(np.linspace(0, 1, numGridNodes), -uLimit[0]*np.ones(numGridNodes), c = 'b', marker = 'x')
ax1.scatter(np.linspace(0, 1, numGridNodes),  uLimit[1]*np.ones(numGridNodes), c = 'orange', marker = 'x')
ax1.scatter(np.linspace(0, 1, numGridNodes), -uLimit[1]*np.ones(numGridNodes), c = 'orange', marker = 'x')
ax1.set_ylabel('control in Nm')
ax1.set_xlabel('normalized time')
ax1.grid()
ax1.set_xlim([0, 1])

plt.show()

# -----------------------------------------------------------------------------

optim.__del__()