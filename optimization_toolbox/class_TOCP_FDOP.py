import numpy as np

from core.control_cubSPL_zeroClamped import Control
from core.Management_FreeDyn import FreeDyn
from core.consistent_boundary_conditions import BC_FDOP
from core.BDF_physicalTime import BDF
from core.adjGrad_wrt_u_and_tF_FDOP import adjGrads
from user_fcts import fcts_User

import core.numerical_differentiation as numDiff

class dataOpt:
    
    def __init__(self, nOptVars,
                 tF, xF):
        
        self.num_opt_vars = nOptVars
        self.num_xF = len(xF)
        
        self.final_time = tF
        self.xF = xF


class Optimization():
    
    def __init__(self,
                      num_optVars, num_ctrls, num_ctrl_gridNodes,
                      tF, xF,
                      path_fds, name_fds,
                      name_ctrlSPL, name_fDu_par,
                      path_FDdll):
        
        self.data_opt = dataOpt(num_optVars, tF, xF)
        
        self.ctrl = Control(num_ctrls, num_ctrl_gridNodes)
        self.FreeDyn = FreeDyn(path_FDdll, path_fds, name_fds, name_ctrlSPL, name_fDu_par)
        self.user_Fcts = fcts_User(self.FreeDyn.nDof, self.data_opt.num_xF, num_ctrls)
        self.adjGrads = adjGrads(self.data_opt, self.FreeDyn, self.ctrl)

        print('class Optimization initialized \n')
# -----------------------------------------------------------------------------
            
    def update_vars_if_changed(self, z):
        
        """ Check if the solution is already computed for z, otherwise reset and recompute """
        
        # Get new values of z
        new_tf = z[0]
        mat_ctrl_gridNodes_new = z[1:].reshape((self.ctrl.num_grid_nodes, self.ctrl.num_ctrls),order='F')
        
        # compare of change
        tf_changed = (new_tf != self.data_opt.final_time)
        u_changed = not np.array_equal(self.ctrl.grid_nodes, mat_ctrl_gridNodes_new)
        
        # reuse or compute solution, only assign and compute if changed
        if tf_changed or u_changed:
            self.data_opt.final_time = new_tf
            self.ctrl.grid_nodes = mat_ctrl_gridNodes_new.copy()
            self.FreeDyn.update_ctrl_gridNodes(self.data_opt, self.ctrl)
            self.FreeDyn.API.reset_for_rerun()
            self.FreeDyn.API.compute_initial_conditions()
            self.FreeDyn.API.solve_until(self.data_opt.final_time) 
            self.FreeDyn.num_time_steps = self.FreeDyn.API.get_num_time_steps()
# -----------------------------------------------------------------------------

    def costFct_J(self, z):
        
        """ Cost functioncal J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        """ Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        J = 0
        
        # compute t = t_f as t_i+1
        self.FreeDyn.API.fetch_states_at_index(self.FreeDyn.num_time_steps-1)
        self.FreeDyn.API.update_state_at_index(self.FreeDyn.num_time_steps-1)   # necessary, if measures are used in get_Lagrangian()
        t_right = self.FreeDyn.API.t
        integrand_right = self.user_Fcts.get_Lagrangian(self.data_opt, self.FreeDyn, self.ctrl, z)
        
        # t_i are computed, t_i+1 are the old values of t_i
        for i in range(self.FreeDyn.num_time_steps-2, -1, -1):
            self.FreeDyn.API.fetch_states_at_index(i)
            self.FreeDyn.API.update_state_at_index(i)   # necessary, if measures are used in get_Lagrangian()
            t_left = self.FreeDyn.API.t
            integrand_left = self.user_Fcts.get_Lagrangian(self.data_opt, self.FreeDyn, self.ctrl, z)

            J += (t_right - t_left) * (integrand_left + integrand_right)
            
            # t_i+1 are the old values of t_i
            t_right = t_left
            integrand_right = integrand_left
            
        J *= 0.5
            
        return J   
# -----------------------------------------------------------------------------
    
    def grad_costFct_J(self, z):
        
        """ Gradient of the cost functioncal J """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_J(self, z)
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        # Returns the gradient by the adjoint method
        return self.adjGrads.adjGrad_J(self.data_opt, self.user_Fcts, self.FreeDyn, self.ctrl, z)        
# -----------------------------------------------------------------------------

    def finalConstr_Phi(self, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        # set t = t_f
        # Phi is evaluted in user_fcts.py
        self.FreeDyn.API.fetch_states_at_index(self.FreeDyn.num_time_steps-1)
        self.FreeDyn.API.update_state_at_index(self.FreeDyn.num_time_steps-1)   # necessary, if measures are used in eval_Phi()
        return self.user_Fcts.eval_Phi(self.data_opt, self.FreeDyn)
# -----------------------------------------------------------------------------
    
    def grad_finalConstr_Phi(self, z):
        
        """ Gradient of the final constraints Phi """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_Phi(self, z)
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)      

        # Returns the gradient by the adjoint method
        return self.adjGrads.adjGrad_Phi(self.data_opt, self.user_Fcts, self.FreeDyn, self.ctrl, z)
# -----------------------------------------------------------------------------