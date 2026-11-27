import numpy as np
from scipy.sparse.linalg import spsolve



class CoreBC():
    
    @classmethod
    def create(cls, FreeDyn, num_xF):
        if FreeDyn.MBS_modeMAT_sparse:
            return SparseBC(FreeDyn, num_xF)
        else:
            return DenseBC(FreeDyn, num_xF)


    def __init__(self, FreeDyn, num_xF):
        
        """ BC for Phi - if necessary """
        if num_xF > 0:
            dim = FreeDyn.nDof + FreeDyn.nConstr
            self.BC_dq_tr = np.zeros((dim, num_xF))   # (dPhi / dq)^T
            self.BC_dv_tr = np.zeros((dim, num_xF))   # (dPhi / dv)^T
            
# -----------------------------------------------------------------------------
    def get_consistent_BC_J(self, BDF, FreeDyn):
        
        FreeDyn.update_sys_mat_for_consistent_BC_J()
        adjP = np.zeros(FreeDyn.nDof)
        adjW = np.zeros(FreeDyn.nDof)

        BDF.set_adj_BC_J(adjP, adjW) 
# -----------------------------------------------------------------------------

    def get_consistent_BC_Phi(self, BDF, UserFcts, FreeDyn):
        
        UserFcts.get_Phi_dq(FreeDyn)    # (dPhi / dq)^T
        UserFcts.get_Phi_dv(FreeDyn)    # (dPhi / dv)^T
        
        self.BC_dq_tr[:FreeDyn.nDof,:] = UserFcts.dPhidq.T    # (dPhi / dq)^T
        self.BC_dv_tr[:FreeDyn.nDof,:] = UserFcts.dPhidv.T    # (dPhi / dv)^T        
        
        FreeDyn.update_sys_mat_for_consistent_BC_Phi()

        WL_tF, PU_tF = self.compute_consistent_BC_Phi(FreeDyn)
        
        BDF.set_adj_BC_Phi(PU_tF[:FreeDyn.nDof, :], WL_tF[:FreeDyn.nDof, :])
# -----------------------------------------------------------------------------
    

class DenseBC(CoreBC):
    
# -----------------------------------------------------------------------------
    def compute_consistent_BC_Phi(self, FreeDyn):        

        coeffMat_W = FreeDyn.SysMat.coeff_mat_BC_adjW
        coeffMat_P = FreeDyn.SysMat.coeff_mat_BC_adjP

        PU_tF = np.linalg.solve(coeffMat_P, self.BC_dv_tr)
        U_tF = PU_tF[FreeDyn.nDof:, :]
        
        self.BC_dq_tr[:FreeDyn.nDof, :] -= (FreeDyn.SysMat.CqvDq.T @ U_tF)
         
        WL_tF = np.linalg.solve(coeffMat_W, self.BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------
    

class SparseBC(CoreBC): 
    
# -----------------------------------------------------------------------------
    def compute_consistent_BC_Phi(self, FreeDyn):        
        
        coeffMat_W = FreeDyn.SysMat.coeff_mat_BC_adjW
        coeffMat_P = FreeDyn.SysMat.coeff_mat_BC_adjP
        
        PU_tF = spsolve(coeffMat_P,self.BC_dv_tr)
        
        U_tF = PU_tF[FreeDyn.nDof:, :]
        self.BC_dq_tr[:FreeDyn.nDof, :] -= (FreeDyn.SysMat.CqvDq.T @ U_tF)
        
        WL_tF = spsolve(coeffMat_W,self.BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------