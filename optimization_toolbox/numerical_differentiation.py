import numpy as np

class numDiff:
    
    def __init__(self):
        
        self.numDiff_stepSize = 1e-6
# -----------------------------------------------------------------------------     
    
    def numGrad_J(self, z):
        
        grad_J = np.zeros(self.num_optVars)
        J = self.objective(z)
        
        for j in range(0, self.num_optVars): 
            z[j] = z[j] + self.numDiff_stepSize
            delta = self.objective(z)
            grad_J[j] = (1/self.numDiff_stepSize) * (delta - J)
            z[j] = z[j] - self.numDiff_stepSize
            
        return grad_J
# -----------------------------------------------------------------------------     
    
    def numGrad_Phi(self, z):
        
        grad_Phi = np.zeros([self.num_xF, self.num_optVars])
        ceq = self.ceq_tF(z)
        
        for j in range(0, self.num_optVars): 
            z[j] = z[j] + self.numDiff_stepSize
            delta = self.ceq_tF(z)
            grad_Phi[:,j] = (1/self.numDiff_stepSize) * (delta - ceq)
            z[j] = z[j] - self.numDiff_stepSize
        
        return grad_Phi
# ----------------------------------------------------------------------------- 