### Required Parameters for the FreeDyn Optimization Toolbox

```python
from optimization_toolbox import OptimCtrl, TimeOptimCtrl, ParamIdent

Opt_TB = <ClassName>(fds_path, fds_name,
                     fds_dForce_dparam_name, fds_meas_name, fds_ctrl_name,
                     num_ctrls, num_ctrl_grid_nodes, num_FD_pars,
                     final_time, final_state,
                     path_FDdll)
```

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
          x) <code>None</code> : &lt;ParamIdent&gt;<br>
          x) <code>Sequence[str]</code> : &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt;
      </td>
    </tr>
    <tr>
      <td>num_ctrls</td>
      <td><code>int</code></td>
      <td>Number of controls<br>
          x) &lt;ParamIdent&gt; : num_ctrls $= 0$<br>
          x) &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt; : num_ctrls $> 0$
      </td>
    </tr>
    <tr>
      <td>num_ctrl_grid_nodes</td>
      <td><code>int</code></td>
      <td>Number of grid nodes per control<br>
          x) &lt;ParamIdent&gt; : num_ctrl_grid_nodes $= 0$<br>
          x) &lt;OptimCtrl&gt; or &lt;TimeOptimCtrl&gt; : num_ctrl_grid_nodes $> 0$ 
      </td>
    </tr>
    <tr>
      <td>num_FD_pars</td>
      <td><code>int</code></td>
      <td>Number of FreeDyn parameters<br>
          x) &lt;ParamIdent&gt; : num_FD_pars $> 0$<br>
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
