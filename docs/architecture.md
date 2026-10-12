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
