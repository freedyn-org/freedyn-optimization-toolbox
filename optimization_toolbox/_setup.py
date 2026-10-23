from __future__ import annotations
from typing import Sequence, Union
import numpy as np

from .toolbox import Toolbox

from .update_optim_vars_if_changed import UpdateOCP, UpdateTOCP, UpdateParamIdent
from .optimization_task import OptimTask
from .FreeDyn_management import FreeDyn
from .control_cubSPL_zeroClamped import Control

from .adjGrad_wrt_u_FDOP import adjGrads_OCP
from .adjGrad_wrt_u_and_tF_FDOP import adjGrads_TOCP
from .adjGrad_wrt_FD_param import adjGrads_Param

# =============================================================================
# Optimal control problem with fixed final time
# =============================================================================
def OptimCtrl(
    fds_path: str, fds_name: str,
    fds_dForce_dparam_name: Sequence[str],
    fds_meas_name: Sequence[str] | None,
    fds_ctrl_name: Sequence[str] | None,
    num_ctrls: int, 
    num_ctrl_grid_nodes: int, 
    num_FD_pars: int, 
    final_time: float, 
    final_state: Union[np.ndarray, None],
    path_FDdll: None | str) -> Toolbox:
    
    num_optVars = num_ctrls * num_ctrl_grid_nodes

    freedyn = FreeDyn(path_FDdll, fds_path, fds_name, 
                      fds_ctrl_name, fds_meas_name, fds_dForce_dparam_name,
                      num_FD_pars)
    ctrl = Control(num_ctrls, num_ctrl_grid_nodes)
    optim_task = OptimTask(UpdateOCP, adjGrads_OCP, freedyn, ctrl, 
                          num_optVars, num_ctrls, final_time, final_state)
    
    
    return Toolbox(freedyn, ctrl, optim_task)

# =============================================================================
# Optimal control problem with free final time
# =============================================================================
def TimeOptimCtrl(
    fds_path: str, fds_name: str,
    fds_dForce_dparam_name: Sequence[str],
    fds_meas_name: Sequence[str] | None,
    fds_ctrl_name: Sequence[str] | None,
    num_ctrls: int, 
    num_ctrl_grid_nodes: int, 
    num_FD_pars: int, 
    final_time: float, 
    final_state: Union[np.ndarray, None],
    path_FDdll: None | str) -> Toolbox:
    
    num_optVars = 1 + num_ctrls * num_ctrl_grid_nodes
    
    freedyn = FreeDyn(path_FDdll, fds_path, fds_name, 
                      fds_ctrl_name, fds_meas_name, fds_dForce_dparam_name,
                      num_FD_pars)
    ctrl = Control(num_ctrls, num_ctrl_grid_nodes)
    optim_task = OptimTask(UpdateTOCP, adjGrads_TOCP, freedyn, ctrl, 
                          num_optVars, num_ctrls, final_time, final_state)
    
    return Toolbox(freedyn, ctrl, optim_task)

# =============================================================================
# Parameter Identification with fixed final time
# =============================================================================
def ParamIdent(
    fds_path: str, fds_name: str,
    fds_dForce_dparam_name: Sequence[str],
    fds_meas_name: Sequence[str] | None,
    fds_ctrl_name: Sequence[str] | None,
    num_ctrls: int, 
    num_ctrl_grid_nodes: int, 
    num_FD_pars: int, 
    final_time: float, 
    final_state: Union[np.ndarray, None],
    path_FDdll: None | str) -> Toolbox:
    
    num_optVars = num_FD_pars

    freedyn = FreeDyn(path_FDdll, fds_path, fds_name, 
                      fds_ctrl_name, fds_meas_name, fds_dForce_dparam_name,
                      num_FD_pars)
    ctrl = Control(num_ctrls, num_ctrl_grid_nodes)
    optim_task = OptimTask(UpdateParamIdent, adjGrads_Param, freedyn, ctrl, 
                          num_optVars, num_FD_pars, final_time, final_state)
    
    
    return Toolbox(freedyn, ctrl, optim_task)