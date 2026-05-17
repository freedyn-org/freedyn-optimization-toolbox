# FreeDyn Optimization Toolbox
**Adjoint-based optimization toolbox for multibody systems using the FreeDyn API**

## Optimal Control Problems in Multibody Dynamics

The objective is to identify the control $\mathbf{u}(t)$ that minimizes the cost functional

$$
\begin{aligned}
J = \int_{t_0}^{t_\mathrm{f}} \mathcal{L} (\mathbf{q}(t),\mathbf{v}(t),\mathbf{u}(t))  \mathrm{d}t
\end{aligned}
$$

in which $\mathcal{L} (\mathbf{q}(t),\mathbf{v}(t),\mathbf{u}(t))$ is the Lagrangian of the optimal control problem, while at final time $t_\mathrm{f}$ possible constraints

$$
\begin{aligned}
\boldsymbol{\phi}(\mathbf{q}(t_\mathrm{f}), \mathbf{v}(t_\mathrm{f}), t_\mathrm{f}) = \mathbf{0}
\end{aligned}
$$

have to be satisfied.

## Features
- **Optimal Control Problems with fixed final time** `class_OCP_MBS.py`
- **Optimal Control Problems with free final time** `class_TOCP_MBS.py`

## Examples
- [Trajectory Tracking/](example_01_two_mass_oscillator_trajectory_tracking/)
- [Minimum Control Effort/](example_02_nonlinear_spring_pendulum_minimum_control_effort/) 
- [Time Optimal Control/](example_03_SCARA_time_optimal_control/) 

## Citation
If you use the FreeDyn Optimization Toolbox in your research, please cite:

```bibtex
@software{freedynOptToolbox2026,
  title = {FreeDyn: Optimization Toolbox},
  author = {FreeDyn Development Team},
  year = {2026},
  url = {https://github.com/freedyn-org/freedyn-optimization-toolbox}
}
```

## Related Papers
Papers demonstrating the toolbox:

P. Zallinger, L. Buchner, W. Steiner, K. Nachbagauer. A Low-Effort Approach to Adjoint Gradient Computation for Optimal Control Problems Using a Multibody Dynamics Simulation Interface. 
ASME International Conference on Multibody Systems, Nonlinear Dynamics, and Control, Houston, Texas, USA, 2026.
