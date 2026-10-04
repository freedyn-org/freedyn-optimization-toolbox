import numpy as np
import math


class fcts_User():
    
    def __init__(self, nState, nXF, nCtrl):
        
        # Allocate matrices for derivatives of Lagrangian: L_q , L_v , L_u
        self.dLdq = np.zeros(nState)                            # do not change 
        self.dLdv = np.zeros(nState)                            # do not change 
        self.dLdu = np.zeros(nCtrl)                             # do not change 
        
        # Allocate matrices for derivatives of final constraints: Phi_q , Phi_v
        if nXF > 0:
            self.dPhidq = np.zeros((nXF, nState))               # do not change 
            self.dPhidv = np.zeros((nXF, nState))               # do not change
        
        print("User functions loaded")
# -----------------------------------------------------------------------------

    def get_Lagrangian(self, dataOpt, FreeDyn, Ctrl, z):
        
        # Lagrangian of the optimization problem: J = \int_{t_0}^{t_f} L dt
        
        ybar = self.get_target_path(FreeDyn.API.t)
        y = FreeDyn.API.Q[7,0] - FreeDyn.API.Q[0,0]   # x2(t) - x1(t)
        delta = y - ybar

        return 0.5 * delta * delta
# -----------------------------------------------------------------------------

    def get_Lagrangian_dq(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdq = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        ybar = self.get_target_path(FreeDyn.API.t)
        y = FreeDyn.API.Q[7,0] - FreeDyn.API.Q[0,0]   # x2(t) - x1(t)
        delta = y - ybar
           
        self.dLdq[0] = - delta
        self.dLdq[7] = delta
        
        return None
# -----------------------------------------------------------------------------

    def get_Lagrangian_dv(self, FreeDyn, z):
        
        # Allocate in __init__ as self.dLdv = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Lagrangian_du(self, dataOpt, FreeDyn, Ctrl, z):
        
        # Allocate in __init__ as self.dLdu = np.zeros(FreeDyn.nDof)
        # If you want to zero all entries, use self.dLdu.fill(0.0)
        # If you want to access an element, use self.dLdu[i] = ...
        # If dLdu = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def eval_Phi(self, FreeDyn, xF):
        
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

    def get_target_path(self, t):
        
        return math.sin(t) + 0.7 * math.sin(math.pi*t) + 0.5 * math.sin(math.sqrt(2)*t)
# -----------------------------------------------------------------------------

    def get_target_path_dt(self, t):
        
        return math.cos(t) + 0.7 * math.pi* math.cos(math.pi*t) + 0.5 * math.sqrt(2)* math.cos(math.sqrt(2)*t)
# -----------------------------------------------------------------------------