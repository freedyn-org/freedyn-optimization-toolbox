import numpy as np
import scipy
from scipy.sparse.linalg import factorized
from scipy.sparse import bmat


class BC_FDOP():
    
    def __init__(self):
        
        """ BC for J """
        if self.FreeDyn.MBS_modeMAT_sparse:   
            self.compute_consistent_BC_J = self.compute_consistent_BC_J_sparse
        else:
            self.compute_consistent_BC_J = self.compute_consistent_BC_J_dense
            
        """ BC for Phi - if necessary """
        if self.num_xF > 0:  
            self.BDF_BC_dq_tr = np.zeros((self.FreeDyn.nDofConstr, self.num_xF))   # (dPhi / dq)^T
            self.BDF_BC_dv_tr = np.zeros((self.FreeDyn.nDofConstr, self.num_xF))   # (dPhi / dv)^T
            
            if self.FreeDyn.MBS_modeMAT_sparse:   
                self.BDF_BC_eyeMat = scipy.sparse.eye(self.FreeDyn.nDof)
                self.compute_consistent_BC_Phi = self.compute_consistent_BC_Phi_sparse
            else:
                self.compute_consistent_BC_Phi = self.compute_consistent_BC_Phi_dense
                self.BDF_BC_eyeMat = np.eye(self.FreeDyn.nDof)
                self.BDF_BC_zeroMat = np.zeros((self.FreeDyn.nConstr, self.FreeDyn.nConstr))   
# -----------------------------------------------------------------------------
    
    def get_consistent_BC_J(self):
        
        self.FreeDyn.slot_MBS_M.update_from_dll()
        self.FreeDyn.slot_MBS_M.apply_to_cached_matrix()
        
        self.compute_consistent_BC_J()
        
        self.BDF_idx_buff = 0
        self.adjP_J_buff[self.BDF_idx_buff, :].fill(0.0)
        self.adjW_J_buff[self.BDF_idx_buff, :].fill(0.0)  
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_dense(self):
        return None 
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_sparse(self):
        return None
# -----------------------------------------------------------------------------
    
    def get_consistent_BC_Phi(self):
        
        self.get_Phi_dq()    # (dPhi / dq)^T
        self.get_Phi_dv()    # (dPhi / dv)^T
        
        self.BDF_BC_dq_tr[:self.FreeDyn.nDof,:] = self.dPhidq.T    # (dPhi / dq)^T
        self.BDF_BC_dv_tr[:self.FreeDyn.nDof,:] = self.dPhidv.T    # (dPhi / dv)^T        
        
        self.FreeDyn.slot_MBS_M.update_from_dll()
        self.FreeDyn.slot_MBS_M.apply_to_cached_matrix()
        self.FreeDyn.slot_MBS_Cq.update_from_dll()
        self.FreeDyn.slot_MBS_Cq.apply_to_cached_matrix()
        self.FreeDyn.slot_MBS_CqvDq.update_from_dll()
        self.FreeDyn.slot_MBS_CqvDq.apply_to_cached_matrix()

        WL_tF, PU_tF = self.compute_consistent_BC_Phi()
        
        self.BDF_idx_buff = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = WL_tF[:self.FreeDyn.nDof, :]     # W_tF
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = PU_tF[:self.FreeDyn.nDof, :]     # P_tF
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_dense(self):        

        coeffMat_W = np.block([[self.BDF_BC_eyeMat, self.FreeDyn.MBS_Cq.T],
                                   [self.FreeDyn.MBS_Cq, self.BDF_BC_zeroMat]])
        
        coeffMat_P = np.block([[self.FreeDyn.MBS_M, self.FreeDyn.MBS_Cq.T],
                               [self.FreeDyn.MBS_Cq, self.BDF_BC_zeroMat]])

        
        PU_tF = np.linalg.solve(coeffMat_P, self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.FreeDyn.nDof:, :]
        
        self.BDF_BC_dq_tr[:self.FreeDyn.nDof, :] -= (self.FreeDyn.MBS_CqvDq.T @ U_tF)
         
        WL_tF = np.linalg.solve(coeffMat_W, self.BDF_BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_sparse(self):        
        
        coeffMat_W_csc = bmat([[self.BDF_BC_eyeMat, self.FreeDyn.MBS_Cq.T],
                               [self.FreeDyn.MBS_Cq, None]], format = 'csc')

        coeffMat_P_csc = bmat([[self.FreeDyn.MBS_M, self.FreeDyn.MBS_Cq.T],
                               [self.FreeDyn.MBS_Cq, None]], format = 'csc')
        
        solve_W = factorized(coeffMat_W_csc)
        solve_P = factorized(coeffMat_P_csc)

        PU_tF = solve_P(self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.FreeDyn.nDof:, :]
        
        self.BDF_BC_dq_tr[:self.FreeDyn.nDof, :] -= (self.FreeDyn.MBS_CqvDq.T @ U_tF)
         
        WL_tF = solve_W(self.BDF_BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------