import numpy as np
import scipy
from scipy.sparse.linalg import factorized


from core.BDF_intOrderOne_physicalTime import BDF_intOrderOne 
from core.BDF_intOrderTwo_physicalTime import BDF_intOrderTwo 
import core.BDF_coeffMat as coeffMat

class BDF(BDF_intOrderOne, BDF_intOrderTwo):
    
    def __init__(self):
        
        # Allocate memory for the coeff. Matrix of the adj Sys
        if self.MBS_modeMAT_sparse:  
            nBDFsys = coeffMat.init_coeffMat_AdjSys_sparse('csc')
        else:
            nBDFsys = coeffMat.init_coeffMat_AdjSys_dense(self)
            
        # Create buffers for fast access
        self.BDF_idx_buff = 0
        num_buffs = 2   # do not change - buffer layout changes require fixes in many places
        self.BDF_diff_tau = np.zeros(num_buffs)
        
        self.adjW_J_buff = np.empty((num_buffs, self.nDof))
        self.adjP_J_buff = np.empty((num_buffs, self.nDof))
        self.BDF_J_buff_M_times_p = np.zeros((num_buffs, self.nDof))
        self.BDF_solVec_J = np.zeros(nBDFsys)
        
        if self.num_xF > 0:
            self.adjW_Phi_buff = np.empty((num_buffs, self.nDof, self.num_xF))
            self.adjP_Phi_buff = np.empty((num_buffs, self.nDof, self.num_xF))
            self.BDF_Phi_buff_M_times_P = np.zeros((num_buffs, self.nDof, self.num_xF))
            self.BDF_solVec_Phi = np.zeros((nBDFsys, self.num_xF))
        
        BDF_intOrderOne.__init__(self)
        BDF_intOrderTwo.__init__(self)
           
        print('class BDF initialized')
        
# =============================================================================
# Update of relevant system matrices in the BDF routine
# =============================================================================

    def update_MBS_SysMat_dll_nonzeros(self):
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_G_tr.update_from_dll()      

    def update_MBS_SysMat(self):
        self.update_MBS_SysMat_dll_nonzeros()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        self.slot_MBS_fv.apply_to_cached_matrix()
        self.slot_MBS_G_tr.apply_to_cached_matrix()
        
# =============================================================================
# Return values of adjVar p at BDF time idx s_n from the buffer
# =============================================================================

    def get_adjVar_p_J(self):
        return self.adjP_J_buff[self.BDF_idx_buff, :]
    
    def get_adjVar_P_Phi(self):
        return self.adjP_Phi_buff[self.BDF_idx_buff,:,:]

# =============================================================================
# Solve the BDF System 
# =============================================================================

    def solve_J_AdjSys_dense(self):
        return np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_J)

    def solve_Phi_AdjSys_dense(self):
        return np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_Phi)   

    def solve_J_AdjSys_sparse(self):
        solve = factorized(self.BDF_spCoeffMat)
        return solve(self.BDF_solVec_J)

    def solve_Phi_AdjSys_sparse(self):
        solve = factorized(self.BDF_spCoeffMat)
        return solve(self.BDF_solVec_Phi)