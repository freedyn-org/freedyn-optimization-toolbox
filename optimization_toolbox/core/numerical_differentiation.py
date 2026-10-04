import numpy as np

# ----------------------------------------------------------------------------- 

def check_grad_J(self, z):
    
    """ Verification of the adjoint gradient via numerical differentiation """
    
    self.update_vars_if_changed(z)   
    adjGrad = self.adjGrads.adjGrad_J(self.data_opt, self.user_Fcts, self.FreeDyn, self.ctrl, z)   # Returns the gradient by the adjoint method
    numGrad = numGrad_J(self, z)  # Returns the gradient by numerical differentiation
    error =  numGrad - adjGrad
        
    return error
# ----------------------------------------------------------------------------- 

def check_grad_Phi(self, z):
    
    """ Verification of the adjoint gradient via numerical differentiation """
    
    self.update_vars_if_changed(z)   
    adjGrad = self.adjGrads.adjGrad_Phi(self.data_opt, self.user_Fcts, self.FreeDyn, self.ctrl, z)   # Returns the gradient by the adjoint method
    numGrad = numGrad_Phi(self, z)  # Returns the gradient by numerical differentiation
    error =  numGrad - adjGrad
        
    return error
# -----------------------------------------------------------------------------

def numGrad_J(self, z):
    
    """ Gradient of the cost functioncal J via numerical differentiation """
    
    grad_J = np.zeros(self.data_opt.num_opt_vars)
    J = self.costFct_J(z)
    
    numDiff_stepSize = 1e-6
    
    for j in range(0, self.data_opt.num_opt_vars): 
        z[j] = z[j] + numDiff_stepSize
        deltaJ = self.costFct_J(z)
        grad_J[j] = (1/numDiff_stepSize) * (deltaJ - J)
        z[j] = z[j] - numDiff_stepSize
        
    return grad_J
# -----------------------------------------------------------------------------     

def numGrad_Phi(self, z):
    
    """ Gradient of the final constraints Phi via numerical differentiation """
    
    grad_Phi = np.zeros([self.data_opt.num_xF, self.data_opt.num_opt_vars])
    Phi = self.finalConstr_Phi(z)
    
    numDiff_stepSize = 1e-6
    
    for j in range(0, self.data_opt.num_opt_vars): 
        z[j] = z[j] + numDiff_stepSize
        deltaPhi = self.finalConstr_Phi(z)
        grad_Phi[:,j] = (1/numDiff_stepSize) * (deltaPhi - Phi)
        z[j] = z[j] - numDiff_stepSize
    
    return grad_Phi
# ----------------------------------------------------------------------------- 