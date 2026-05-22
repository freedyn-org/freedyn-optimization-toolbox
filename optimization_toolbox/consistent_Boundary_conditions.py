import numpy as np
import scipy
from scipy.sparse.linalg import factorized
from scipy.sparse import bmat


class BC_consistent():
    
    def __init__(self):
        
        self.BDF_BC_dq_tr = np.zeros((self.nDofConstr, self.num_xF))   # (dPhi / dq)^T
        self.BDF_BC_dv_tr = np.zeros((self.nDofConstr, self.num_xF))   # (dPhi / dv)^T
        
        if self.MBS_modeMAT_sparse:   
            self.BDF_BC_eyeMat = scipy.sparse.eye(self.nDof)
            self.compute_consistent_BC_J_FDOP = self.compute_consistent_BC_J_FDOP_sparse
            self.compute_consistent_BC_Phi_FDOP = self.compute_consistent_BC_Phi_FDOP_sparse

        else:
            self.compute_consistent_BC_J_FDOP = self.compute_consistent_BC_J_FDOP_dense
            self.compute_consistent_BC_Phi_FDOP = self.compute_consistent_BC_Phi_FDOP_dense
            self.BDF_BC_eyeMat = np.eye(self.nDof)
            self.BDF_BC_zeroMat = np.zeros((self.nConstr, self.nConstr))
# -----------------------------------------------------------------------------
    
    def get_consistent_BC_J_FDOP(self):
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        
        self.compute_consistent_BC_J_FDOP()
        
        self.adjP_J_buff[self.BDF_idx_buff, :].fill(0.0)
        self.adjW_J_buff[self.BDF_idx_buff, :].fill(0.0)  
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_FDOP_dense(self):
        return None 
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_FDOP_sparse(self):
        return None
# -----------------------------------------------------------------------------
    
    def get_consistent_BC_Phi_FDOP(self):
        
        self.get_Phi_dq()    # (dPhi / dq)^T
        self.get_Phi_dv()    # (dPhi / dv)^T
        
        self.BDF_BC_dq_tr[:self.nDof,:] = self.dPhidq.T    # (dPhi / dq)^T
        self.BDF_BC_dv_tr[:self.nDof,:] = self.dPhidv.T    # (dPhi / dv)^T        
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        
        self.BDF_idx_buff = 0
        
        
        WL_tF, PU_tF = self.compute_consistent_BC_Phi_FDOP()
        
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = WL_tF[:self.nDof, :]     # W_tF
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = PU_tF[:self.nDof, :]     # P_tF

        return None
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_FDOP_dense(self):        

        coeffMat_W = np.block([[self.BDF_BC_eyeMat, self.MBS_Cq.T],
                                   [self.MBS_Cq, self.BDF_BC_zeroMat]])
        
        
        coeffMat_P = np.block([[self.MBS_M, self.MBS_Cq.T],
                               [self.MBS_Cq, self.BDF_BC_zeroMat]])

        
        PU_tF = np.linalg.solve(coeffMat_P, self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.nDof:, :]
        
        
        self.BDF_BC_dq_tr[:self.nDof, :] -= (self.MBS_CqvDq.T @ U_tF)
         
        WL_tF = np.linalg.solve(coeffMat_W, self.BDF_BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_sparse(self):        
        
        coeffMat_W_csc = bmat([[self.BDF_BC_eyeMat, self.MBS_Cq.T],
                               [self.MBS_Cq, None]], format = 'csc')
        
        
        coeffMat_P_csc = bmat([[self.MBS_M, self.MBS_Cq.T],
                               [self.MBS_Cq, None]], format = 'csc')
        
        
        solve_W = factorized(coeffMat_W_csc)
        solve_P = factorized(coeffMat_P_csc)


        PU_tF = solve_P(self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.nDof:, :]
        
        
        self.BDF_BC_dq_tr[:self.nDof, :] -= (self.MBS_CqvDq.T @ U_tF)
         
        WL_tF = solve_W(self.BDF_BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------