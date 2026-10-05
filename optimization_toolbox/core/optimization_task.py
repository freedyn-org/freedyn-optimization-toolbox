import numpy as np

from user_fcts import fcts_User

from core.adjGrad_wrt_u_FDOP import adjGrads_OCP
from core.adjGrad_wrt_u_and_tF_FDOP import adjGrads_TOCP
from core.adjGrad_wrt_FD_param import adjGrads_Param


class opt_task():
    
    def __init__(self, task, FreeDyn, ctrl,
                 nOptVars,
                 tF, xF):
        
        self.num_opt_vars = nOptVars
        self.num_xF = len(xF)
        self.final_time = tF
        self.xF = xF
        
        if task == "OCP":
            self.adjGrads = adjGrads_OCP(self, FreeDyn, ctrl)
            self.user_Fcts = fcts_User(FreeDyn.nDof, self.num_xF, ctrl.num_ctrls)
        elif task == "TOCP":
            self.adjGrads = adjGrads_TOCP(self, FreeDyn, ctrl)
            self.user_Fcts = fcts_User(FreeDyn.nDof, self.num_xF, ctrl.num_ctrls)
        elif task == "Parameter":
            self.adjGrads = adjGrads_Param(self, FreeDyn, FreeDyn.num_FD_pars)
            self.user_Fcts = fcts_User(FreeDyn.nDof, self.num_xF, FreeDyn.num_FD_pars)

# -----------------------------------------------------------------------------    
    
    def costFct_J(self, FreeDyn, Ctrl, z):
        
        """ Cost functioncal J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
        
        """ Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        J = 0
        
        # compute t = t_f as t_i+1 
        FreeDyn.API.fetch_states_at_index(FreeDyn.num_time_steps-1)
        FreeDyn.API.update_state_at_index(FreeDyn.num_time_steps-1)   # necessary, if measures are used in get_Lagrangian()
        t_right = FreeDyn.API.t
        integrand_right = self.user_Fcts.get_Lagrangian(self, FreeDyn, Ctrl, z)
        
        # t_i are computed, t_i+1 are the old values of t_i
        for i in range(FreeDyn.num_time_steps-2, -1, -1):
            FreeDyn.API.fetch_states_at_index(i)
            FreeDyn.API.update_state_at_index(i)   # necessary, if measures are used in get_Lagrangian()
            t_left = FreeDyn.API.t
            integrand_left = self.user_Fcts.get_Lagrangian(self, FreeDyn, Ctrl, z)

            J += (t_right - t_left) * (integrand_left + integrand_right)
            
            # t_i+1 are the old values of t_i
            t_right = t_left
            integrand_right = integrand_left
            
        J *= 0.5
        
        return J
# -----------------------------------------------------------------------------
    
    def finalConstr_Phi(self, FreeDyn, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # set t = t_f
        # Phi is evaluted in user_fcts.py
        FreeDyn.API.fetch_states_at_index(FreeDyn.num_time_steps-1)
        FreeDyn.API.update_state_at_index(FreeDyn.num_time_steps-1)   # necessary, if measures are used in eval_Phi()
        return self.user_Fcts.eval_Phi(self, FreeDyn)
# -----------------------------------------------------------------------------