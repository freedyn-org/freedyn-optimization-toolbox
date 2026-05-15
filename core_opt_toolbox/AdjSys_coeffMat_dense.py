import numpy as np

class CoeffMat():
    
    def __init__(self):
        
        nBDFsys = self.nDof+2*self.nConstr

        self.init_coeffMat_AdjSys_dense(nBDFsys)

        return nBDFsys
        
# -----------------------------------------------------------------------------

    def init_coeffMat_AdjSys_dense(self, nBDFsys):

        self.BDF_coeffMat = np.zeros((nBDFsys, nBDFsys))
        self.BDF_coeffMat_view_11 = self.BDF_coeffMat[:self.nDof, :self.nDof] 
        self.BDF_coeffMat_view_12 = self.BDF_coeffMat[:self.nDof, self.nDof:self.nDofConstr]
        self.BDF_coeffMat_view_13 = self.BDF_coeffMat[:self.nDof, self.nDofConstr:]
        self.BDF_coeffMat_view_21 = self.BDF_coeffMat[self.nDof:self.nDofConstr, :self.nDof]
        self.BDF_coeffMat_view_22 = self.BDF_coeffMat[self.nDof:self.nDofConstr, self.nDof:self.nDofConstr]
        self.BDF_coeffMat_view_23 = self.BDF_coeffMat[self.nDof:self.nDofConstr, self.nDofConstr:]
        self.BDF_coeffMat_view_31 = self.BDF_coeffMat[self.nDofConstr:, :self.nDof]

        return None

# -----------------------------------------------------------------------------

    def get_coeffMat_AdjSys_dense(self, eta0, eta0_inv):
        
        self.BDF_coeffMat_view_11[:] = eta0 * self.MBS_M - eta0_inv * self.MBS_G_tr.T - self.MBS_fv.T
        self.BDF_coeffMat_view_12[:] = -eta0_inv * self.MBS_CqvDq.T - self.MBS_Cq.T
        self.BDF_coeffMat_view_13[:] = -eta0_inv * self.MBS_Cq.T
        self.BDF_coeffMat_view_21[:] = self.MBS_Cq @ self.MBS_G_tr.T
        self.BDF_coeffMat_view_22[:] = self.MBS_Cq @ self.MBS_CqvDq.T
        self.BDF_coeffMat_view_23[:] = self.MBS_Cq @ self.MBS_Cq.T
        self.BDF_coeffMat_view_31[:] = self.MBS_Cq

        return None

# -----------------------------------------------------------------------------