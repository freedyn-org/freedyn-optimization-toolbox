import numpy as np

import scipy
from scipy.sparse import bmat

class spCoeffMat_3x3():
    
    def __init__(self):
        
        nBDFsys = self.nDof+2*self.nConstr
        
        self.BDF_coeffMat_csr = None
        
        return nBDFsys
        
# -----------------------------------------------------------------------------

    def get_coeffMat_AdjSys_3x3_sparse(self, eta0, eta0_inv):
        
        A11 = eta0 * self.MBS_M - eta0_inv * self.MBS_G_tr.T - self.MBS_fv.T
        A12 = - eta0_inv * self.MBS_CqvDq.T - self.MBS_Cq.T
        A13 = - eta0_inv * self.MBS_Cq.T
        
        A21 = self.MBS_Cq @ self.MBS_G_tr.T
        A22 = self.MBS_Cq @ self.MBS_CqvDq.T
        A23 = self.MBS_Cq @ self.MBS_Cq.T

        
        self.BDF_spCoeffMat = bmat([[A11, A12, A13],
                                      [A21, A22, A23],
                                      [self.MBS_Cq, None, None]], format = 'csc')

        return None

# -----------------------------------------------------------------------------