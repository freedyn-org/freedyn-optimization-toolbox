# Setup Custom Optimization 

- `model.fds` → [Model](setup_guide.md#Model)
- `user_fcts.py` → [Optimization Task](setup_guide.md#Optimization-Task)
- `main.py` → [Setup Optimization](setup_guide.md#Setup-Optimization)

## Model

The dynamical model underlying the optimization is provided as FreeDyn multibody system `*.fds`.

- Bodies
- Constraints
- Forces/Torques
- Measures
- Data Object
	- Parameter
	- Spline
		- Interpolation type: ` akima ` [akima spline](https://en.wikipedia.org/wiki/Akima_spline)
		- Interpolation type: ` zccspl ` [zero clamped cubic spline](https://doi.org/10.1007/s11044-023-09898-5)

### Additional Requirements for Optimal Control Problems: `<OptimCtrl> or <TimeOptimCtrl>`
To compute the adjoint gradient with respect to a control, the partial derivative of the force vector with respect to the control is required. 
This derivative is not computed by default in a forward solution of the model. With the following additions, this derivative is provided for gradient computation:

1) Create FreeDyn Parameter for each control with Value = 0.0
```text 
DataObject Parameter
	ID = ID_key
	Name = u_par
	InitialValue = 0.000000
EndDataObject
```

2) Call FreeDyn Parameter in the Amplitude of Force/Torque Expression
```text 
PARVAL[u_par] + SPL[SIMTIME[], u_Dach]
```

3) Usage of the Spline Type `zccspl`
```text
InterpolationType = zccspl
```
Note: The identical control discretization must be used in the optimization and FreeDyn, otherwise errors in the gradient computation occur.
Within the FreeDyn Optimization Toolbox, the spline type `zccspl` is used for computing the adjoint gradients.


## Optimization Task

The objective is to identify the optimization variables $\mathbf{z}$ that minimizes the cost functional

$$
\begin{aligned}
J(\mathbf{z}) = \int_{t_0}^{t_\mathrm{f}} \mathcal{L} (\mathbf{q}(t),\mathbf{v}(t),\mathbf{u}(t))  \mathrm{d}t
\end{aligned}
$$

in which $\mathcal{L} (\mathbf{q}(t),\mathbf{v}(t),\mathbf{u}(t))$ is the Lagrangian of the optimization problem, while at final time $t_\mathrm{f}$ possible constraints

$$
\begin{aligned}
\boldsymbol{\phi}(\mathbf{q}(t_\mathrm{f}), \mathbf{v}(t_\mathrm{f}), t_\mathrm{f}) = \mathbf{0}
\end{aligned}
$$

have to be satisfied.

$$
\begin{aligned}
\mathbf{q}(t) \dots \text{Vector of redundant generalized coordinates}\\
\mathbf{v}(t) \dots \text{Vector of redundant generalized velocities}\\
\mathbf{u}(t) \dots \text{Vector of controls (forces, torques, etc.)}
\end{aligned}
$$

The states of the multibody system, i.e., $\mathbf{q}$, $\dot{\mathbf{q}}$, $\ddot{\mathbf{q}}$ and~$\matLambda$, can be accessed via the time index~$t_i$. By fetching the states, the associated physical time instance~$t$ is also updated. In FreeDyn, the rotational DOF are described by Euler parameters, which affects the dimension~$\mathbb{R}^n$ and $\mathbb{R}^c$. 

The only quantities, which have to be implemented by the user, are~$\OptTarget$ and~$\BCend$, as well as their partial derivatives with respect to $\mathbf{q}$, $\mathbf{v}$ and $\mathbf{u}$.
By defining measures in FreeDyn, arbitrary displacements, velocities, angles, etc. can be determined. These can be used in FreeDyn, but are also accessible via the API.

<table>
  <thead>
    <tr>
      <th>Derivative</th>
      <th>Allocated Buffer</th>
      <th>Function in user_fcts.py</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$\mathcal{L}_{\mathbf{q}}$</td>
      <td><code>self.dLdq</code></td>
      <td>Path of <code>*.fds</code></td>
    </tr>
	  <tr>
      <td>$\mathcal{L}_{\mathbf{v}}$</td>
      <td><code>self.dLdv</code></td>
      <td>Path of <code>*.fds</code></td>
    </tr>
	  <tr>
      <td>$\mathcal{L}_{\mathbf{u}}$</td>
      <td><code>self.dLdu</code></td>
      <td>Path of <code>*.fds</code></td>
    </tr>
    <tr>
      <td>$\boldsymbol{\phi}_{\mathbf{q}}$</td>
      <td><code>self.dPhidq</code></td>
      <td>Name of <code>*.fds</code> (without file type)</td>
    </tr>
    <tr>
      <td>$\boldsymbol{\phi}_{\mathbf{v}}$</td>
      <td><code>self.dPhidv</code></td>
      <td>Final time of the optimization problem</td>
    </tr>
  </tbody>
</table>

```python
        # Allocate matrices for derivatives of lagrangian: L_q , L_v , L_u
        # Allocate matrices for derivatives of final constraints: Phi_q , Phi_v
        
        print("User functions loaded")
# -----------------------------------------------------------------------------

    def get_lagrangian(self, dataOpt, FreeDyn, Ctrl, z):
        
        # lagrangian of the optimization problem: J = \int_{t_0}^{t_f} L dt
        
        u = Ctrl.get_u(FreeDyn.API.t / dataOpt.final_time)

        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_dq(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_dv(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdv = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_du(self, dataOpt, FreeDyn, Ctrl, z):
        
        # Allocate in __init__ as self.dLdu = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdu.fill(0.0)
        # If you want to access an element, use self.dLdu[i] = ...
        # If dLdu = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def eval_Phi(self, dataOpt, FreeDyn):
        
        # Final constraints: Phi(t_f) = 0
        # If no final constraints are imposed, then only use "return None"
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dq(self, FreeDyn):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dv(self, FreeDyn):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------
```

## Setup Optimization
1) Add path of FreeDyn Optimization Toolbox
```python
path_main = Path(__file__).resolve().parent
path_folder_optim_toolbox = str(path_main.parent.parent)
sys.path.insert(0, path_folder_optim_toolbox)
```

2) Define the Parameters of FreeDyn Optimization Toolbox

[`Parameters of FreeDyn Optimization Toolbox`](docs/architecture.md#required-parameters-for-the-freedyn-optimization-toolbox)

3) Initialize FreeDyn Optimization Toolbox
```python
from optimization_toolbox import OptimCtrl, TimeOptimCtrl, ParamIdent

Opt_TB = <ClassName>(fds_path, fds_name,
                     fds_dForce_dparam_name, fds_meas_name, fds_ctrl_name,
                     num_ctrls, num_ctrl_grid_nodes, num_FD_pars,
                     final_time, final_state,
                     path_FDdll)
```

4) Carry out optimization with third-party optimization toolbox, e.g., SciPy
```python
import scipy as sp

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
```

5) Post-processing of the optimization result
```python 
Opt_TB.new_opt_vars(res.x)                              # Update optimization variables, rerun simulation
num_steps = Opt_TB.FreeDyn.num_time_steps
final_time = Opt_TB.OptimTask.final_time
time = np.zeros(num_steps)                              
u = np.zeros((num_ctrls, num_steps))                    # optimal control
q = np.zeros((Opt_TB.FreeDyn.nDof, num_steps))          # redundant generalized coordinates
qD = np.zeros((Opt_TB.FreeDyn.nDof, num_steps))         # redundant generalized velocities
    
for i in range(num_steps-1, -1, -1): 
   Opt_TB.FreeDyn.fetch_and_update_states_at_index(i)
   time[i] = Opt_TB.FreeDyn.API.t
   u[:,i] = Opt_TB.Ctrl.get_u(tau[i]/final_time)
   q[:,i] = Opt_TB.FreeDyn.API.Q[:, 0]
   qD[:,i] = Opt_TB.FreeDyn.API.Qd[:, 0]
```
