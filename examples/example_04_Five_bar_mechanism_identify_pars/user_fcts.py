import numpy as np
import math


class UserFunctions():
    
    def __init__(self, num_q, num_xF, num_pars):
        
        # Allocate matrices for derivatives of lagrangian: L_q , L_v , L_u
        self.dLdq = np.zeros(num_q)                                            # do not change 
        self.dLdv = np.zeros(num_q)                                            # do not change 
        self.dLdpars = np.zeros(num_pars)                                      # do not change
        
        # Allocate matrices for derivatives of final constraints: Phi_q , Phi_v
        if num_xF > 0:
            self.dPhidq = np.zeros((num_xF, num_q))                            # do not change 
            self.dPhidv = np.zeros((num_xF, num_q))                            # do not change
        
        print("User functions loaded")
# -----------------------------------------------------------------------------

    def get_lagrangian(self, dataOpt, FreeDyn, Ctrl, z):
        
        # lagrangian of the optimization problem: J = \int_{t_0}^{t_f} L dt
        
        # r_P2 = np.zeros(2)
        
        # for i, par in enumerate(self.name_fDmeas):
        #     r_P2[i] = self.API.get_measure_value(self.name_fDmeas[i])
        
        
        # r_P2_init = np.array([0.0, -2.0])
        # delta = r_P2 - r_P2_init
        
        
        r_COM1 = np.zeros(2)
        
        r_COM1[0] = FreeDyn.API.Q[0,0]
        r_COM1[1] = FreeDyn.API.Q[1,0] 
        
        
        r_COM1_init = np.array([-1.0, -0.5])
        delta = r_COM1 - r_COM1_init

        
        
        return 0.5 * np.dot(delta, delta)
# -----------------------------------------------------------------------------

    def get_lagrangian_dq(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        r_COM1 = np.zeros(2)
        
        r_COM1[0] = FreeDyn.API.Q[0,0]
        r_COM1[1] = FreeDyn.API.Q[1,0] 
        
        
        r_COM1_init = np.array([-1.0, -0.5])
        delta = r_COM1 - r_COM1_init
           
        self.dLdq[0] = delta[0]
        self.dLdq[1] = delta[1]
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_dv(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdv = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_dpars(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdpars = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdpars.fill(0.0)
        # If you want to access an element, use self.dLdpars[i] = ...
        # If dLdpars = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def eval_Phi(self, dataOpt, FreeDyn):
        
        # Final constraints: Phi(t_f) = 0
        # If no final constraints are imposed, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dq(self, FreeDyn):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dv(self, FreeDyn):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------
#
        """        Define here your own functions        """
#
# -----------------------------------------------------------------------------