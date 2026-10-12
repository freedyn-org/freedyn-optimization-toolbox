| Files of FreeDyn Optimization Toolbox | Tasks                                                                 |
|---------------------------------------|-----------------------------------------------------------------------|
| _setup.py                             | Integration of methods based on the optimization task                 |
| toolbox.py                            | Interface to third-party optimization toolbox                         |
| Management_FreeDyn.py                 | Interface to FreeDyn & its API                                        |
| control_cubSPL_zeroClamped.py         | Provides cubic spline interpolation for control                       |
| optimization_task.py                  | Defines the optimization problem                                      |
| update_optim_vars_if_changed.py       | Reuse or compute solution in FreeDyn                                  |
| adjGrad_wrt_u_FDOP.py                 | Computes adjoint gradients - Optimal Control Problem                  |
| adjGrad_wrt_u_and_tF_FDOP.py          | Computes adjoint gradients - Time-Optimal Control Problem             |
| adjGrad_wrt_FD_param.py               | Computes adjoint gradients - Parameter-Identification                 |
| consistent_boundary_conditions.py     | Provides consistent boundary conditions for the adjoint systems       |
| BDF_core.py                           | Backward Differentiation Formula - common functions and memory buffer |
| BDF_dense_physical_time.py            | BDF order 1 and 2 for dense system matrices                           |
| BDF_sparse_physical_time.py           | BDF order 1 and 2 for sparse system matrices                          |


| Example-Dependent Files | Tasks                                                                           |
|-------------------------|---------------------------------------------------------------------------------|
| main_*.py               | Provides the FreeDyn Optimization Toolbox the necessary parameters              |
| user_fcts.py            | Definition of cost functional and final constraints including their derivatives |

### Required Parameters for the FreeDyn Optimization Toolbox
```python
fds_path: str,                           # Path of *.fds
fds_name: str,                           # Name of *.fds (without file typ)
fds_dForce_dparam_name: Sequence[str],   # FreeDyn param for df / dparam
fds_meas_name: Sequence[str] | None,     # FreeDyn measures: required, if measures are used in user_fcts.py
fds_ctrl_name: Sequence[str] | None,     # FreeDyn control splines: required for OptimCtrl or TimeOptimCtrl
num_ctrls: int,                          # Number of controls
num_ctrl_grid_nodes: int,                # Number of grid nodes per control
num_FD_pars: int,                        # Number of FreeDyn parameters 
final_time: float,                       # Final time
final_state: Sequence[str] | None,       # Constraints at final time
path_FDdll: None | str                   # Path to FreeDyn dll:
                                         # x) Use NONE: pip install freedyn
                                         # x) Use str: FreeDyn Python bindings are download from GitHub without installing
```
