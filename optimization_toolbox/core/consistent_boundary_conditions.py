import numpy as np
import scipy
from scipy.sparse.linalg import factorized
from scipy.sparse import bmat



class CoreBC():
    
    def __init__(self, dataOpt, FreeDyn):
        
        """ BC for Phi - if necessary """
        if dataOpt.num_xF > 0:  
            self.BC_dq_tr = np.zeros((FreeDyn.nDofConstr, dataOpt.num_xF))   # (dPhi / dq)^T
            self.BC_dv_tr = np.zeros((FreeDyn.nDofConstr, dataOpt.num_xF))   # (dPhi / dv)^T
            
# -----------------------------------------------------------------------------
    def get_consistent_BC_J(self, BDF, FreeDyn):
        
        FreeDyn.update_sys_mat_for_consistent_BC_J()
        
        self.compute_consistent_BC_J()
        
        BDF.BDF_idx_buff = 0
        BDF.adjP_J_buff[0, :].fill(0.0)
        BDF.adjW_J_buff[0, :].fill(0.0)  
# -----------------------------------------------------------------------------
    def get_consistent_BC_Phi(self, BDF, UserFcts, FreeDyn):
        
        UserFcts.get_Phi_dq(FreeDyn)    # (dPhi / dq)^T
        UserFcts.get_Phi_dv(FreeDyn)    # (dPhi / dv)^T
        
        self.BC_dq_tr[:FreeDyn.nDof,:] = UserFcts.dPhidq.T    # (dPhi / dq)^T
        self.BC_dv_tr[:FreeDyn.nDof,:] = UserFcts.dPhidv.T    # (dPhi / dv)^T        
        
        FreeDyn.update_sys_mat_for_consistent_BC_Phi()

        WL_tF, PU_tF = self.compute_consistent_BC_Phi(FreeDyn)
        
        BDF.BDF_idx_buff = 0
        BDF.adjW_Phi_buff[0, :, :] = WL_tF[:FreeDyn.nDof, :]     # W_tF
        BDF.adjP_Phi_buff[0, :, :] = PU_tF[:FreeDyn.nDof, :]     # P_tF
# -----------------------------------------------------------------------------
    

class DenseBC(CoreBC):
    
    def __init__(self, dataOpt, FreeDyn):
        
        """ BC for Phi - if necessary """
        if dataOpt.num_xF > 0:  
            self.BC_eyeMat = np.eye(FreeDyn.nDof)
            self.BC_zeroMat = np.zeros((FreeDyn.nConstr, FreeDyn.nConstr)) 
            
        CoreBC.__init__(self, dataOpt, FreeDyn)
    
# -----------------------------------------------------------------------------
        
    def compute_consistent_BC_J(self):
        return None 
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi(self, FreeDyn):        

        coeffMat_W = np.block([[self.BC_eyeMat, FreeDyn.MBS_Cq.T],
                                   [FreeDyn.MBS_Cq, self.BC_zeroMat]])
        
        coeffMat_P = np.block([[FreeDyn.MBS_M, FreeDyn.MBS_Cq.T],
                               [FreeDyn.MBS_Cq, self.BC_zeroMat]])

        
        PU_tF = np.linalg.solve(coeffMat_P, self.BC_dv_tr)
        U_tF = PU_tF[FreeDyn.nDof:, :]
        
        self.BC_dq_tr[:FreeDyn.nDof, :] -= (FreeDyn.MBS_CqvDq.T @ U_tF)
         
        WL_tF = np.linalg.solve(coeffMat_W, self.BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------
    

class SparseBC(CoreBC): 
    
    def __init__(self, dataOpt, FreeDyn):
        
        """ BC for Phi - if necessary """
        if dataOpt.num_xF > 0:  
            eye_dense = np.eye(FreeDyn.nDof)
            self.BC_eyeMat = scipy.sparse.csr_matrix(eye_dense)
        
        CoreBC.__init__(self, dataOpt, FreeDyn)
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J(self):
        return None
    
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi(self, FreeDyn):        
        
        coeffMat_W_csc = bmat([[self.BC_eyeMat, FreeDyn.MBS_Cq.T],
                               [FreeDyn.MBS_Cq, None]], format = 'csc')

        coeffMat_P_csc = bmat([[FreeDyn.MBS_M, FreeDyn.MBS_Cq.T],
                               [FreeDyn.MBS_Cq, None]], format = 'csc')
        
        solve_W = factorized(coeffMat_W_csc)
        solve_P = factorized(coeffMat_P_csc)

        PU_tF = solve_P(self.BC_dv_tr)
        U_tF = PU_tF[FreeDyn.nDof:, :]
        
        self.BC_dq_tr[:FreeDyn.nDof, :] -= (FreeDyn.MBS_CqvDq.T @ U_tF)
         
        WL_tF = solve_W(self.BC_dq_tr)

        return WL_tF, PU_tF
# -----------------------------------------------------------------------------