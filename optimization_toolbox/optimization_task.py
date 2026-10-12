import numpy as np

from user_fcts import UserFunctions

class OptimTask():
    
    def __init__(self, UpdateOptimVars, AdjGrad, FreeDyn, Ctrl,
                 nOptVars, npars,
                 tF, xF):
        
        self.num_opt_vars = nOptVars
        self.num_xF = len(xF)
        self.final_time = tF
        self.xF = xF
        
        self.UpdateOptimVars = UpdateOptimVars()
        self.adjGrads = AdjGrad(self, FreeDyn, Ctrl)
        self.UserFcts = UserFunctions(FreeDyn.nDof, self.num_xF, npars)

# =============================================================================
#  Cost functional J
# =============================================================================   
    
    def cost_fct_J(self, FreeDyn, Ctrl, z):
        
        self.UpdateOptimVars.compute_or_reuse(self, FreeDyn, Ctrl, z)
        
        """ Cost functional J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
        
        """ Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        J = 0
        
        # compute t = t_f as t_i+1
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        t_right = FreeDyn.API.t
        integrand_right = self.UserFcts.get_lagrangian(self, FreeDyn, Ctrl, z)
        
        # t_i are computed, t_i+1 are the old values of t_i
        for i in range(FreeDyn.num_time_steps-2, -1, -1):
            FreeDyn.fetch_and_update_states_at_index(i)
            t_left = FreeDyn.API.t
            integrand_left = self.UserFcts.get_lagrangian(self, FreeDyn, Ctrl, z)

            J += (t_right - t_left) * (integrand_left + integrand_right)
            
            # t_i+1 are the old values of t_i
            t_right = t_left
            integrand_right = integrand_left
            
        J *= 0.5
        
        return J
    
# -----------------------------------------------------------------------------

    def adj_grad_J(self, FreeDyn, Ctrl, z):
        
        self.UpdateOptimVars.compute_or_reuse(self, FreeDyn, Ctrl, z)
        
        """ Gradient of the cost functioncal J by the adjoint method """
        
        return self.adjGrads.adjGrad_J(self, self.UserFcts, FreeDyn, Ctrl, z)
# -----------------------------------------------------------------------------

    def num_grad_J(self, FreeDyn, Ctrl, z):
        
        """ Gradient of the cost functioncal J via numerical differentiation """
        
        grad_J = np.zeros(self.num_opt_vars)
        J = self.cost_fct_J(FreeDyn, Ctrl, z)
        
        num_diff_step_size = 1e-6
        
        for j in range(0, self.num_opt_vars): 
            z[j] = z[j] + num_diff_step_size
            deltaJ = self.cost_fct_J(FreeDyn, Ctrl, z)
            grad_J[j] = (1/num_diff_step_size) * (deltaJ - J)
            z[j] = z[j] - num_diff_step_size
            
        return grad_J
    
# ----------------------------------------------------------------------------- 

    def check_grad_J(self, FreeDyn, Ctrl, z):
        
        """ Verification of the adjoint gradient via numerical differentiation """
        adjGrad = self.adj_grad_J(FreeDyn, Ctrl, z)
        numGrad = self.num_grad_J(FreeDyn, Ctrl, z)
        error =  numGrad - adjGrad
            
        return error
    
# =============================================================================
#  Final constraints Phi
# ============================================================================= 
    
    def final_constr_eq_Phi(self, FreeDyn, Ctrl, z):
        
        self.UpdateOptimVars.compute_or_reuse(self, FreeDyn, Ctrl, z)
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # set t = t_f
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        
        # Phi is evaluted in UserFcts.py
        return self.UserFcts.eval_Phi(self, FreeDyn)

# -----------------------------------------------------------------------------    

    def adj_grad_Phi(self, FreeDyn, Ctrl, z):
        
        self.UpdateOptimVars.compute_or_reuse(self, FreeDyn, Ctrl, z)
        
        """ Gradient of the final constraints Phi by the adjoint method """
        
        return self.adjGrads.adjGrad_Phi(self, self.UserFcts, FreeDyn, Ctrl, z)
# -----------------------------------------------------------------------------

    def num_grad_Phi(self, FreeDyn, Ctrl, z):
        """ Gradient of the final constraints Phi via numerical differentiation """
        
        grad_Phi = np.zeros([self.num_xF, self.num_opt_vars])
        Phi = self.final_constr_eq_Phi(FreeDyn, Ctrl, z)
        
        num_diff_step_size = 1e-6
        
        for j in range(0, self.num_opt_vars): 
            z[j] = z[j] + num_diff_step_size
            deltaPhi = self.final_constr_eq_Phi(FreeDyn, Ctrl, z)
            grad_Phi[:,j] = (1/num_diff_step_size) * (deltaPhi - Phi)
            z[j] = z[j] - num_diff_step_size
        
        return grad_Phi
# ----------------------------------------------------------------------------- 

    def check_grad_Phi(self, FreeDyn, Ctrl, z):
        
        """ Verification of the adjoint gradient via numerical differentiation """
        adjGrad = self.adj_grad_Phi(FreeDyn, Ctrl, z)
        numGrad = self.num_grad_Phi(FreeDyn, Ctrl, z)
        error =  numGrad - adjGrad
            
        return error
# -----------------------------------------------------------------------------