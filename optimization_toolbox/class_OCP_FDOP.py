import numpy as np
import freedyn as fd

from core.control_cubSPL_zeroClamped import Control
from core.Management_FreeDyn import FreeDyn
from core.consistent_boundary_conditions import BC_FDOP
from core.BDF_physicalTime import BDF
from core.adjGrad_wrt_u_FDOP import adjGrads
from user_fcts import fcts_User

import core.numerical_differentiation as numDiff


class Optimization(Control, FreeDyn, BC_FDOP, BDF, adjGrads, fcts_User):
    
    def __init__(self,
                      num_optVars, num_ctrls, num_ctrl_gridNodes,
                      tF, xF,
                      path_fds, name_fds,
                      name_ctrlSPL, name_fDu_par,
                      path_FDdll):
        
        self.num_optVars = num_optVars
        self.tF = tF
        self.ctrl_gridNodes = None
        self.xF = xF
        self.num_xF = len(xF)

        Control.__init__(self, num_ctrls, num_ctrl_gridNodes)
        FreeDyn.__init__(self, path_FDdll, path_fds, name_fds, name_ctrlSPL, name_fDu_par)
        BC_FDOP.__init__(self)
        BDF.__init__(self)
        adjGrads.__init__(self)
        fcts_User.__init__(self)

        print('class Optimization initialized \n')
# -----------------------------------------------------------------------------
        
    def __del__(self):
        FreeDyn.__del__(self)     
# -----------------------------------------------------------------------------
            
    def update_vars_if_changed(self, z):
        
        """ Check if the solution is already computed for z, otherwise reset and recompute """
        
        # Get new values of z
        mat_ctrl_gridNodes_new = z.reshape((self.num_ctrl_gridNodes, self.num_ctrls),order='F')              

        # compare of change
        u_changed = not np.array_equal(self.ctrl_gridNodes, mat_ctrl_gridNodes_new)

        # reuse or compute solution, only assign and compute if changed
        if u_changed:
            self.ctrl_gridNodes = mat_ctrl_gridNodes_new.copy()
            self.update_ctrl_gridNodes()
            self.fd_model.reset_for_rerun()
            self.fd_model.compute_initial_conditions()
            self.fd_model.solve_until(self.tF) 
            self.num_time_steps = self.fd_model.get_num_time_steps()  
# -----------------------------------------------------------------------------

    def costFct_J(self, z):
        
        """ Cost functioncal J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
         
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        """ Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        J = 0
        
        # compute t = t_f as t_i+1 
        self.fd_model.fetch_states_at_index(self.num_time_steps-1)
        self.fd_model.update_state_at_index(self.num_time_steps-1)   # necessary, if measures are used in get_Lagrangian()
        t_right = self.fd_model.t
        integrand_right = self.get_Lagrangian(z)
        
        # t_i are computed, t_i+1 are the old values of t_i
        for i in range(self.num_time_steps-2, -1, -1):
            self.fd_model.fetch_states_at_index(i)
            self.fd_model.update_state_at_index(i)   # necessary, if measures are used in get_Lagrangian()
            t_left = self.fd_model.t
            integrand_left = self.get_Lagrangian(z)

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
        return self.adjGrad_J(z)            
# -----------------------------------------------------------------------------

    def finalConstr_Phi(self, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        # set t = t_f
        # Phi is evaluted in user_fcts.py
        self.fd_model.fetch_states_at_index(self.num_time_steps-1)
        self.fd_model.update_state_at_index(self.num_time_steps-1)   # necessary, if measures are used in eval_Phi()
        return self.eval_Phi()
# -----------------------------------------------------------------------------
    
    def grad_finalConstr_Phi(self, z):
        
        """ Gradient of the final constraints Phi """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_Phi(self, z)
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.update_vars_if_changed(z)
        
        # Returns the gradient by the adjoint method
        return self.adjGrad_Phi(z)
# -----------------------------------------------------------------------------