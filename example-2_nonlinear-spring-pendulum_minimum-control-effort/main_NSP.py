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
pathFDdll = 'C:\\VRoboCoop\\Programme\\FreeDyn\\Release_1.0.5\\FreeDyn-win-x64_MD\\freedyn.dll'

# Path to FreeDyn API """
pathFDApi = '..\\..\\freedyn\\bindings\\python'
sys.path.insert(0, pathFDApi)

# Path to core_opt_toolbox
bib_path = '..\\core_opt_toolbox'
sys.path.insert(0, bib_path)

# Define path and name of *.fds - without file typ!
path_fds = Path(__file__).resolve().parent
name_fds = 'OptCtrl_NonlinearSpringPendulum'


"""  Choose OCP Problem """
# Final time tF is fixed: Optimal control problem with/without final constraints Phi
from class_OCP_MBS import Optimization

# Final time tF is free: Time-optimal control problem with/without final constraints Phi
# from class_OCP_MBS import Optimization     


""" Define the names of ctrl splines """
# derivative of sum of external forces w.r.t. parameter given as string
nameCtrlSpline = ["uDach_x", "uDach_y", "uDach_z"]


""" Define the names of the parameters for dfdu """
# derivative of sum of external forces w.r.t. parameter given as string
nameParFdu = ["fdu_x","fdu_y","fdu_z"]


""" Define controls """
numControls = 3           # number of controls
numGridNodes = 10        # number of grid nodes per control
uDachInit = np.zeros(numGridNodes*numControls)


""" Define final state of the MBS system """
tF = 5             # final time
xF = np.array([2,-10,-4,0,0,0])   # final constraints, if no constraints are used, set = np.array([])


""" Define initial values for optimization variables z0"""
z0 = uDachInit      
numOptVar = len(z0)


optim = Optimization(numOptVar, numControls, numGridNodes,
                     tF, xF,
                     path_fds, name_fds,
                     nameCtrlSpline, nameParFdu,
                     pathFDdll)

optFtol = 1e-8
optEps = 1e-8
options = {'disp': True, 'iprint': 2, 'ftol': optFtol, 'eps':optEps, 'maxiter': 5}
constraints = {'type':'eq', 'fun':optim.ceq_tF, 'jac':optim.get_grad_Phi}


profiler = Profiler()
profiler.start()
countStart = time.perf_counter()
res = sp.optimize.minimize(fun         = optim.objective,                    # cost function
                           x0          = z0,                                 # initial values
                           method      = 'SLSQP',                            # optimization method
                           jac         = optim.get_grad_J,                   # gradient of cost function
                           #bounds      = bounds,                             # lower and upper bounds
                           constraints = constraints,                        # non-linear constraints
                           options     = options                             # optimization options
                           )
countEnd = time.perf_counter()
profiler.stop()
profiler.open_in_browser("speedscope")  # Öffnet direkt in speedscope.app


timeComp = countEnd - countStart
print(f"Zeitdauer Optimierung: {timeComp} s")
#
# -----------------------------------------------------------------------------
out_dt = 0.01
optim.change_outputDeltaT_in_fds(out_dt)
optim.update_vars_if_changed(res.x)
# -----------------------------------------------------------------------------

uInit = np.zeros((numControls, optim.dyn_numTimeSteps))
u = np.zeros((numControls, optim.dyn_numTimeSteps))
dyn_t = np.zeros(optim.dyn_numTimeSteps)
dyn_q = np.zeros((optim.nDof, optim.dyn_numTimeSteps))
dyn_qD = np.zeros((optim.nDof, optim.dyn_numTimeSteps))
spring_l = np.zeros(optim.dyn_numTimeSteps)

for i in range(optim.dyn_numTimeSteps-1, -1, -1): 
   optim.fd_model.fetch_states_at_index(i)
   dyn_t[i] = optim.fd_model.get_time_at_index(i) 
   
   dyn_q[:,i] = optim.fd_model.Q[:, 0]
   dyn_qD[:,i] = optim.fd_model.Qd[:, 0]
   spring_l[i] = optim.fd_model.get_measure_value("l")
   
   uInit[:,i] = optim.get_u_for_GridNodes(dyn_t[i]/optim.tF, uDachInit)
   u[:,i] = optim.get_u_for_GridNodes(dyn_t[i]/optim.tF, optim.uDach)  
   

tau = dyn_t/optim.tF

# -----------------------------------------------------------------------------

# plots
# using the variable axs for multiple Axes
matplotlib.rcParams.update({'font.size': 15})
f = plt.figure(figsize=(10,10))

# plot relative motion
ax1 = f.add_subplot(2,2, 1)
ax1.plot(dyn_t, dyn_q[0,:], c = 'blue', linewidth = 2)
ax1.plot(dyn_t, dyn_q[1,:], c = 'green', linewidth = 2)
ax1.plot(dyn_t, dyn_q[2,:], c = 'darkorange', linewidth = 2)
ax1.scatter(np.array([tF,tF,tF]), xF[0:3], c = 'r', marker = 'x')
ax1.set_xlabel('Time in s')
ax1.set_ylabel('Position in m')
ax1.grid()

# plot relative veloctiy
ax2 = f.add_subplot(2,2, 2)
ax2.plot(dyn_t, dyn_qD[0,:], c = 'blue', linewidth = 2)
ax2.plot(dyn_t, dyn_qD[1,:], c = 'green', linewidth = 2)
ax2.plot(dyn_t, dyn_qD[2,:], c = 'darkorange', linewidth = 2)
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
ax4.plot(dyn_t, spring_l, c = 'blue', linewidth = 2)
ax4.set_ylabel('Spring length in m')
ax4.set_xlabel('Time in s')
ax4.grid()

plt.show()
#
# # -----------------------------------------------------------------------------

optim.__del__()
