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
        
        self.compute_consistent_BC_J()
        
        BDF.init_buff_indices()
        BDF.buff_adjP_J[BDF.buff_idx1, :].fill(0.0)
        BDF.buff_adjW_J[BDF.buff_idx1, :].fill(0.0)  
# -----------------------------------------------------------------------------
    def get_consistent_BC_Phi(self, BDF, UserFcts, FreeDyn):
        
        UserFcts.get_Phi_dq(FreeDyn)    # (dPhi / dq)^T
        UserFcts.get_Phi_dv(FreeDyn)    # (dPhi / dv)^T
        
        self.BC_dq_tr[:FreeDyn.nDof,:] = UserFcts.dPhidq.T    # (dPhi / dq)^T
        self.BC_dv_tr[:FreeDyn.nDof,:] = UserFcts.dPhidv.T    # (dPhi / dv)^T        
        
        FreeDyn.update_sys_mat_for_consistent_BC_Phi()

        WL_tF, PU_tF = self.compute_consistent_BC_Phi(FreeDyn)
        
        BDF.init_buff_indices()
        BDF.buff_adjW_Phi[BDF.buff_idx1, :, :] = WL_tF[:FreeDyn.nDof, :]     # W_tF
        BDF.buff_adjP_Phi[BDF.buff_idx1, :, :] = PU_tF[:FreeDyn.nDof, :]     # P_tF
# -----------------------------------------------------------------------------
    

class DenseBC(CoreBC):
    
# -----------------------------------------------------------------------------
        
    def compute_consistent_BC_J(self):
        return None 
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
    
    def compute_consistent_BC_J(self):
        return None
    
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