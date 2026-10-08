import numpy as np
import math


class UserFunctions():
    
    def __init__(self, num_q, num_xF, num_ctrl):
        
        # Allocate matrices for derivatives of lagrangian: L_q , L_v , L_u
        self.dLdq = np.zeros(num_q)                                            # do not change 
        self.dLdv = np.zeros(num_q)                                            # do not change 
        self.dLdu = np.zeros(num_ctrl)                                         # do not change 
        
        # Allocate matrices for derivatives of final constraints: Phi_q , Phi_v
        if num_xF > 0:
            self.dPhidq = np.zeros((num_xF, num_q))                            # do not change 
            self.dPhidv = np.zeros((num_xF, num_q))                            # do not change 

        print("User functions loaded")
# -----------------------------------------------------------------------------

    def get_lagrangian(self, dataOpt, FreeDyn, Ctrl, z):
        
        # lagrangian of the optimization problem: J = \int_{t_0}^{t_f} L dt
        
        L = 1

        return L
# -----------------------------------------------------------------------------

    def get_lagrangian_dq(self, FreeDyn, z):
        
        # Allocated in __init__ as self.dLdq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_dv(self, FreeDyn, z):
        
        # Allocated in __init__ as self.dLdv = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_lagrangian_du(self, dataOpt, FreeDyn, Ctrl, z):
        
        # Allocated in __init__ as self.dLdu = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdu.fill(0.0)
        # If you want to access an element, use self.dLdu[i] = ...
        # If dLdu = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def eval_Phi(self, dataOpt, FreeDyn):
        
        # Final constraints: \Phi(t_f) = 0
        # If no final constraints are imposed, then only use "return None"
        
        x = np.concatenate([FreeDyn.API.Q[14:16, 0],     # x_{TCP}
                            FreeDyn.API.Qd[14:16, 0]])   # \dot{x}_{TCP}
    
        
        return x - dataOpt.xF
# -----------------------------------------------------------------------------

    def get_Phi_dq(self, FreeDyn):
        
        # Allocated in __init__ as self.dPhidq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        self.dPhidq[0,14] = 1
        self.dPhidq[1,15] = 1
        
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dv(self, FreeDyn):
        
        # Allocated in __init__ as self.dPhidv = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dPhidv.fill(0.0)
        # If you want to access an element, use self.dPhidv[i,j] = ...
        # If dPhidv = 0, then only use "return None"
        
        self.dPhidv[2,14] = 1
        self.dPhidv[3,15] = 1
        
        return None
# -----------------------------------------------------------------------------
#
        """        Define here your own functions        """
#
# -----------------------------------------------------------------------------
