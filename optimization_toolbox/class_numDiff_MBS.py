# Author: Philipp Zallinger
# Date: 09.09.2025

import numpy as np

class numDiff:
    
    def __init__(self):
                
        self.step_fd = 1e-6
    
    
    def numGrad_J(self, z):
        
        grad_J = np.zeros(self.numOptVar)
        J = self.objective(z)
        for j in range(0, self.numOptVar): 
            z[j] = z[j] + self.step_fd
            grad_J[j] = (1/self.step_fd) * (self.objective(z) - J)
            z[j] = z[j] - self.step_fd
            
       
        return grad_J
    
    
    def numGrad_Phi(self, z):
        
        grad_Phi = np.zeros([self.num_xF, self.numOptVar])
        
        ceq = self.ceq_tF(z)
        
        for j in range(0, self.numOptVar): 
            z[j] = z[j] + self.step_fd
            grad_Phi[:,j] = (1/self.step_fd) * (self.ceq_tF(z) - ceq)
            z[j] = z[j] - self.step_fd
        
        return grad_Phi