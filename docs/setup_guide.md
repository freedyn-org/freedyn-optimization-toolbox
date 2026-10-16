# Setup Custom Optimization 

- `model.fds`
- `main.py`
- `user_fcts.py`

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

## Setup Optimization

To work with the FreeDyn~API in Python, the basic commands are required. Besides the path to the solver input file, the path to the FreeDyn API needs to be specified.
