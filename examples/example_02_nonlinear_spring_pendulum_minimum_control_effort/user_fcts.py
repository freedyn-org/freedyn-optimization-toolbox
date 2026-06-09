import numpy as np
import math


class fcts_User():
    
    def __init__(self):
        
        # Allocate matrices for derivatives of Lagrangian: L_q , L_v , L_u
        self.dLdq = np.zeros(self.nDof)                         # do not change 
        self.dLdv = np.zeros(self.nDof)                         # do not change 
        self.dLdu = np.zeros(self.num_ctrls)                    # do not change 
        
        # Allocate matrices for derivatives of final constraints: Phi_q , Phi_v
        if self.num_xF > 0:
            self.dPhidq = np.zeros((self.num_xF, self.nDof))    # do not change 
            self.dPhidv = np.zeros((self.num_xF, self.nDof))    # do not change
        
        print("User functions loaded")
# -----------------------------------------------------------------------------

    def get_Lagrangian(self, z):
        
        # Lagrangian of the optimization problem: J = \int_{t_0}^{t_f} L dt
        
        u = self.get_u(self.fd_model.t / self.tF)

        return 0.5 * np.dot(u, u)
# -----------------------------------------------------------------------------

    def get_Lagrangian_dq(self, z):
        
        # Allocate in __init__ as self.dLdq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdq.fill(0.0)
        # If you want to access an element, use self.dLdq[i] = ...
        # If dLdq = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Lagrangian_dv(self, z):
        
        # Allocate in __init__ as self.dLdv = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdv.fill(0.0)
        # If you want to access an element, use self.dLdv[i] = ...
        # If dLdv = 0, then only use "return None"
        
        return None
# -----------------------------------------------------------------------------

    def get_Lagrangian_du(self, z):
        
        # Allocate in __init__ as self.dLdu = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dLdu.fill(0.0)
        # If you want to access an element, use self.dLdu[i] = ...
        # If dLdu = 0, then only use "return None"
        
        self.dLdu = self.get_u(self.fd_model.t / self.tF).T
        
        return None
# -----------------------------------------------------------------------------

    def eval_Phi(self):
        
        # Final constraints: Phi(t_f) = 0
        # If no final constraints are imposed, then only use "return None"
        
        x = np.concatenate([self.fd_model.Q[0:3,0],     # q - translational DOFs
                            self.fd_model.Qd[0:3,0]])   # \dot{q} - transl. DOFs
        
        return x - self.xF
# -----------------------------------------------------------------------------

    def get_Phi_dq(self):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        self.dPhidq[0,0] = 1
        self.dPhidq[1,1] = 1
        self.dPhidq[2,2] = 1
        
        return None
# -----------------------------------------------------------------------------

    def get_Phi_dv(self):
        
        # Allocate in __init__ as self.dPhidq = np.zeros(self.nDof)
        # If you want to zero all entries, use self.dPhidq.fill(0.0)
        # If you want to access an element, use self.dPhidq[i,j] = ...
        # If dPhidq = 0, then only use "return None"
        
        self.dPhidv[3,0] = 1
        self.dPhidv[4,1] = 1
        self.dPhidv[5,2] = 1
        
        return None
# -----------------------------------------------------------------------------
#
        """        Define here your own functions        """
#
# -----------------------------------------------------------------------------
