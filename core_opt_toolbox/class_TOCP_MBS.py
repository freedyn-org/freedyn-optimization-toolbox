import numpy as np


from class_Control import Control
from class_FreeDyn import FreeDyn
from class_BDF_physicalTime import BDF
from class_adjGrad_wrt_uDach_and_tF_MBS import adjGrads
from class_numDiff_MBS import numDiff
from user_fcts import fcts_User


class Optimization(Control, FreeDyn, BDF, adjGrads, numDiff, fcts_User):
    
    def __init__(self,
                      numOptVar, numControls, numGridNodes,
                      tF, xF,
                      path_fds, name_fds,
                      nameCtrlSpline, nameParFdu,
                      pathFDdll):
        
        
        self.numOptVar = numOptVar
        self.tF = tF
        self.uDach = None
        
        
        self.xF = xF
        self.num_xF = len(xF)
        
        
        Control.__init__(self, numControls, numGridNodes)
        FreeDyn.__init__(self, path_fds, name_fds, nameCtrlSpline, nameParFdu, pathFDdll)
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
        mat_uDach_new = z[1:].reshape((self.numNodes, self.numCtrls),order='F')
            
        
        # compare of change
        tf_changed = (new_tf != self.tF)
        u_changed = not np.array_equal(self.uDach, mat_uDach_new)
        
        
        # Only assign if changed
        if tf_changed:
            self.tF = new_tf
            self.change_tF_in_fds()   # triggers a new computation in FreeDyn
            
            if not u_changed:  # txt file is written with physical time -> tf changed this
                self.write_ctrl_dataSPL()  # triggers a new computation in FreeDyn
            
            
            
        if u_changed:
            self.uDach = mat_uDach_new.copy()
            self.write_ctrl_dataSPL()  # triggers a new computation in FreeDyn
        
            
        # reuse or compute solution
        if self.dyn_numTimeSteps == 0:
            self.exec_FreeDyn()  
        
        self.init_ctrl_for_new_optIT()
        
        return None      

# -----------------------------------------------------------------------------

    def objective(self, z):
        
        self.update_vars_if_changed(z)
                    
        return self.eval_J(z)
    
# -----------------------------------------------------------------------------

    def eval_J(self, z):
       
        J = 0
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        integrand_right = self.get_LagrangianOCP(tRight)
        
        for i in range(self.dyn_numTimeSteps-2, -1, -1):
            
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            
            integrand_left = self.get_LagrangianOCP(tLeft)

            J += (tRight - tLeft) * (integrand_left + integrand_right)
            
            
            tRight = tLeft
            integrand_right = integrand_left
            
            
        J *= 0.5
            
        
        return J
    
# -----------------------------------------------------------------------------
    
    def get_grad_J(self, z):
        
        self.update_vars_if_changed(z)   
        
        # grad_J = self.adjGrad_J_BDFthenGrad(z)        
        grad_J = self.adjGrad_J_singleBDFstep(z)
        
        
        """ Numerischer Gradient - all """
        #numGrad_J = self.numGrad_J(z)
        #error =  numGrad_J - grad_J
        
        return grad_J

# -----------------------------------------------------------------------------

    def ceq_tF(self, z):

        self.update_vars_if_changed(z)
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)

        return self.eval_Phi()

# -----------------------------------------------------------------------------
    
    def get_grad_Phi(self, z):
        
        self.update_vars_if_changed(z)
        
        # grad_Phi = self.adjGrad_Phi_BDFthenGrad(z)
        grad_Phi = self.adjGrad_Phi_singleBDFstep(z)
        
        """ Numerischer Gradient - all """
        # numGrad_Phi = self.numGrad_Phi(z)
        # error = numGrad_Phi - grad_Phi
        
        return grad_Phi
    
# -----------------------------------------------------------------------------