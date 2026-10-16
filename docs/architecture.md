### Example-Dependent Files

| File name    | Tasks                                                                           |
|--------------|---------------------------------------------------------------------------------|
| main_*.py    | Provides the FreeDyn Optimization Toolbox the necessary parameters              |
| user_fcts.py | Definition of cost functional and final constraints including their derivatives |

### Files of FreeDyn Optimization Toolbox

| File name                         | Tasks                                                                 |
|-----------------------------------|-----------------------------------------------------------------------|
| _setup.py                         | Integration of methods based on the optimization task                 |
| toolbox.py                        | Interface to third-party optimization toolbox                         |
| Management_FreeDyn.py             | Interface to FreeDyn & its API                                        |
| control_cubSPL_zeroClamped.py     | Provides cubic spline interpolation for control                       |
| optimization_task.py              | Defines the optimization problem                                      |
| update_optim_vars_if_changed.py   | Reuse or compute solution in FreeDyn                                  |
| adjGrad_wrt_u_FDOP.py             | Computes adjoint gradients - Optimal Control Problem                  |
| adjGrad_wrt_u_and_tF_FDOP.py      | Computes adjoint gradients - Time-Optimal Control Problem             |
| consistent_boundary_conditions.py | Provides consistent boundary conditions for the adjoint systems       |
| BDF_core.py                       | Backward Differentiation Formula - common functions and memory buffer |
| BDF_dense_physical_time.py        | BDF order 1 and 2 for dense system matrices                           |
| BDF_sparse_physical_time.py       | BDF order 1 and 2 for sparse system matrices                          |

### Required Parameters for the FreeDyn Optimization Toolbox

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fds_path</td>
      <td><code>str</code></td>
      <td>Path of <code>*.fds</code></td>
    </tr>
    <tr>
      <td>fds_name</td>
      <td><code>str</code></td>
      <td>Name of <code>*.fds</code> (without file type)</td>
    </tr>
    <tr>
      <td>fds_dForce_dparam_name</td>
      <td><code>Sequence[str]</code></td>
      <td>FreeDyn param for df / dparam</td>
    </tr>
    <tr>
      <td>fds_meas_name</td>
      <td><code>None</code><br> 
          <code>Sequence[str]</code>
      </td>
      <td>FreeDyn measures<br>
          x) <code>None</code> : No measures are used in <code>user_fcts.py</code><br>
          x) <code>Sequence[str]</code> : Measures are used in <code>user_fcts.py</code>
      </td>
    </tr>
    <tr>
      <td>fds_ctrl_name</td>
      <td><code>None</code><br> 
          <code>Sequence[str]</code>
      </td>
      <td>FreeDyn control splines<br>
          x) <code>Sequence[str]</code> : &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt;
      </td>
    </tr>
    <tr>
      <td>num_ctrls</td>
      <td><code>int</code></td>
      <td>Number of controls<br>
          x) &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt; : num_ctrls $> 0$
      </td>
    </tr>
    <tr>
      <td>num_ctrl_grid_nodes</td>
      <td><code>int</code></td>
      <td>Number of grid nodes per control<br>
          x) &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt; : num_ctrl_grid_nodes $> 0$ 
      </td>
    </tr>
    <tr>
      <td>num_FD_pars</td>
      <td><code>int</code></td>
      <td>Number of FreeDyn parameters<br>
          x) &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt; : num_FD_pars $= 0$ 
      </td>
    </tr>
    <tr>
      <td>final_time</td>
      <td><code>float</code></td>
      <td>Final time of the optimization problem</td>
    </tr>
    <tr>
      <td>final_state</td>
      <td><code>None</code><br> 
          <code>numpy.ndarray</code>
      </td>
      <td>Possible constraints at final time $\boldsymbol{\phi}$<br>
          x) <code>None</code> : No $\boldsymbol{\phi}$ nor target final_state<br>
          x) <code>numpy.ndarray</code> : Assign target final_state for each $\boldsymbol{\phi}$ in <code>user_fcts.py</code>
      </td>
    </tr>
    <tr>
      <td>path_FDdll</td>
      <td><code>None</code><br> 
          <code>str</code>
      </td>
      <td>Path to <code>freedyn.dll</code> <br>
          x) <code>None</code> : pip install freedyn<br>
          x) <code>str</code> : FreeDyn Python bindings are download without installing
      </td>
    </tr>
  </tbody>
</table>
