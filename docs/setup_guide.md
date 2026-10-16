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

Masses and moments of inertia are assigned for each body. Joints can be assigned between bodies or with ground by locking the corresponding translational and rotational DOF. Forces or torques can be defined either directly in FreeDyn or by external subroutines in the form of dynamic link libraries~(DLL). Note that the identical discretization of the control has to be used in FreeDyn as well as in the optimization, otherwise errors in the gradient evaluation occur. FreeDyn provides different spline types for discretizing continuous variables, e.g., the control.

In force or torque elements, the command~\texttt{SPL[query_point, data_object_spline]} returns the interpolated value at the query point based on the spline data object defined in FreeDyn. In the spline data object, either \texttt{AKISPL} or \texttt{ZCCSPL} can be chosen. The first is the spline type Akima~\cite{Akima1970} and the second is the spline type often used in optimal control problems as well, as e.g., in~\cite{Lichtenecker2023}. In the case of the FDOP, the latter spline type has to be used in FreeDyn in order to obtain the discretization of~$\mathbf{u}$ according to Eq.~(\ref{eq_ctrl_Daniel}). \par

Amplitude of Force/Torque Expression: 
```text 
PARVAL[u_par] + SPL[SIMTIME[], u_Dach]
```


Once the model in FreeDyn is complete, the solver input file (*.fds) is available for the FreeDyn API.

A different method is required for $\mathbf{f}_{\mathbf{u}}$. For each of the $m$ controls, a FreeDyn parameter with zero value is defined. These are evaluated within the force or torque elements type generic. Thus, the derivative of $\mathbf{f}$ with respect to each of these parameters can be computed. In case of $m > 1$, these individual derivatives are combined column-wise in ascending order, i.e., $\mathbf{f}_{\mathbf{u}} = \left[ \mathbf{f}_{u_1}, \dots, \mathbf{f}_{u_m} \right]$.

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
