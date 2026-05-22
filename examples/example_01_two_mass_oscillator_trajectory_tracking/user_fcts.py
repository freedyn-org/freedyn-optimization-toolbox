import numpy as np
import math


class fcts_User():
    
    def __init__(self):
        
        """   Allocate   L_q   L_v   L_u   Phi_q   Phi_v   """
        self.dLdq = np.zeros(self.nDof)                         # do not change 
        self.dLdv = np.zeros(self.nDof)                         # do not change 
        self.dLdu = np.zeros(self.num_ctrls)                     # do not change 
        
        if self.num_xF > 0:
            self.dPhidq = np.zeros((self.num_xF, self.nDof))    # do not change 
            self.dPhidv = np.zeros((self.num_xF, self.nDof))    # do not change
        
        
        print("User functions loaded")
    
# -----------------------------------------------------------------------------

    def get_LagrangianOCP(self, z):
        
        ybar = self.get_target_path(self.fd_model.t)
        delta = self.fd_model.Q[7,0] - self.fd_model.Q[0,0] - ybar

        return 0.5 * delta * delta

# -----------------------------------------------------------------------------

    def get_LagrangianOCP_dq(self, z):
        
        # Allocate in __init__ as self.dLdq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        ybar = self.get_target_path(self.fd_model.t)
        delta = self.fd_model.Q[7,0] - self.fd_model.Q[0,0] - ybar
           
        self.dLdq[0] = - delta
        self.dLdq[7] = delta
        
        return None

# -----------------------------------------------------------------------------

    def get_LagrangianOCP_dv(self, z):
        
        # Allocate in __init__ as self.dLdv = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None

# -----------------------------------------------------------------------------

    def get_LagrangianOCP_du(self, z):
        
        # Allocate in __init__ as self.dLdu = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdu.fill(0.0)
        # If you want to access an element, use self.dLdu[i] = ...
        # If dLdu = 0, then only use "return None"
        
        return None
    
# -----------------------------------------------------------------------------

    def eval_Phi(self):
        
        return None

# -----------------------------------------------------------------------------

    def get_Phi_dq(self):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None

# -----------------------------------------------------------------------------

    def get_Phi_dv(self):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        return None

# -----------------------------------------------------------------------------
    """ Define here your own functions """
# -----------------------------------------------------------------------------

    def get_target_path(self, t):
        
        return math.sin(t) + 0.7 * math.sin(math.pi*t) + 0.5 * math.sin(math.sqrt(2)*t)
        

# -----------------------------------------------------------------------------

    def get_target_path_dt(self, t):
        
        return math.cos(t) + 0.7 * math.pi* math.cos(math.pi*t) + 0.5 * math.sqrt(2)* math.cos(math.sqrt(2)*t)

# -----------------------------------------------------------------------------