import numpy as np
import freedyn as fd

from control_zeroClamped_cubSPL import Control
from Management_FreeDyn import FreeDyn
from BDF_physicalTime import BDF
from adjGrad_wrt_u_and_tF_FDOP import adjGrads
from numDiff_MBS import numDiff
from user_fcts import fcts_User


class Optimization(Control, FreeDyn, BDF, adjGrads, numDiff, fcts_User):
    
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
        BDF.__init__(self)
        numDiff.__init__(self)
        adjGrads.__init__(self)
        fcts_User.__init__(self)

        print('class Optimization initialized \n')
# -----------------------------------------------------------------------------
        
    def __del__(self):
        FreeDyn.__del__(self)
# -----------------------------------------------------------------------------
            
    def update_vars_if_changed(self, z):

        new_tf = z[0]
        mat_ctrl_gridNodes_new = z[1:].reshape((self.num_ctrl_gridNodes, self.num_ctrls),order='F')
        
        # compare of change
        tf_changed = (new_tf != self.tF)
        u_changed = not np.array_equal(self.ctrl_gridNodes, mat_ctrl_gridNodes_new)

        if tf_changed or u_changed:
            self.tF = new_tf
            self.ctrl_gridNodes = mat_ctrl_gridNodes_new.copy()
            self.update_ctrl_gridNodes()
            self.fd_model.reset_for_rerun()
            self.exec_FreeDyn()  
        
        return None      
# -----------------------------------------------------------------------------

    def objective(self, z):
        
        self.update_vars_if_changed(z)
        return self.eval_J(z)
# -----------------------------------------------------------------------------

    def eval_J(self, z):
       
        J = 0
        
        self.fd_model.fetch_states_at_index(self.num_time_steps-1)
        tRight = self.fd_model.t
        integrand_right = self.get_LagrangianOCP(z)
        
        for i in range(self.num_time_steps-2, -1, -1):
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.t
            integrand_left = self.get_LagrangianOCP(z)

            J += (tRight - tLeft) * (integrand_left + integrand_right)
            
            tRight = tLeft
            integrand_right = integrand_left
            
        J *= 0.5
            
        return J   
# -----------------------------------------------------------------------------
    
    def get_grad_J(self, z):
        
        self.update_vars_if_changed(z)   
        grad_J = self.adjGrad_J(z)        

        """ Numerischer Gradient - all """
        #numGrad_J = self.numGrad_J(z)
        #error =  numGrad_J - grad_J
        
        return grad_J
# -----------------------------------------------------------------------------

    def ceq_tF(self, z):

        self.update_vars_if_changed(z)
        self.fd_model.fetch_states_at_index(self.num_time_steps-1)
        return self.eval_Phi()
# -----------------------------------------------------------------------------
    
    def get_grad_Phi(self, z):
        
        self.update_vars_if_changed(z)
        grad_Phi = self.adjGrad_Phi(z)
        
        """ Numerischer Gradient - all """
        # numGrad_Phi = self.numGrad_Phi(z)
        # error = numGrad_Phi - grad_Phi
        
        return grad_Phi
# -----------------------------------------------------------------------------