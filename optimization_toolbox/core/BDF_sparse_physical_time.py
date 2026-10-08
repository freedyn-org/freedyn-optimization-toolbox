import numpy as np
import scipy
from scipy.sparse import bmat
from scipy.sparse.linalg import factorized

from core.BDF_core import CoreBDF


class SparseBDF(CoreBDF):
    
    def __init__(self, dataOpt, FreeDyn):
        nBDFsys = self.init_adj_sys_coeff_mat(FreeDyn, 'csc')
        partition_solVec_1 = FreeDyn.nDof
        partition_solVec_2 = 2*FreeDyn.nDof      
        CoreBDF.__init__(self, dataOpt, FreeDyn, nBDFsys, partition_solVec_1, partition_solVec_2)
        
        return None
    
# =============================================================================
# BDF System - Matrix layout sparse
# =============================================================================

    def init_adj_sys_coeff_mat(self, FreeDyn, format_mat):
        
        """ Allocate the coefficient matrix and create maps of the individual blocks """
        nBDFsys = 2 * (FreeDyn.nDof + FreeDyn.nConstr)
        
        #FreeDyn.update_sys_mat_for_BDF()
        FreeDyn.API.compute_initial_conditions()
        eye_dense = np.eye(FreeDyn.nDof)
        eyeMat_sp = scipy.sparse.csr_matrix(eye_dense)
        dummy_M = FreeDyn.slot_MBS_M.sp_mat.copy()
        dummy_fv = FreeDyn.slot_MBS_fv.sp_mat.copy()
        dummy_M.data.fill(1.0)
        dummy_fv.data.fill(1.0)        
        sumA22 = dummy_M + dummy_fv.T
    
        offset = 0
        offset, A11_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, eyeMat_sp)  
        offset, A12_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_G_tr, transpose=True) 
        offset, A13_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_CqvDq, transpose=True)
        offset, A14_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_Cq, transpose=True) 
        offset, A21_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, eyeMat_sp)
        offset, A22_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, sumA22)        
        offset, A23_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_Cq, transpose=True)
        offset, A32_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_Cq)  
        offset, A41_idx = self.idx_temp_init_adj_sys_coeff_mat(offset, FreeDyn.MBS_Cq)
        
        layout = [[A11_idx, A12_idx, A13_idx, A14_idx],
                  [A21_idx, A22_idx, A23_idx, None],
                  [None, A32_idx, None, None],
                  [A41_idx, None, None, None]]
        
        # Bulid Coeff Mat as csc sparse matrix
        self.coeff_mat = bmat(layout, format=format_mat).astype(np.float64)
        
        # Mapping
        self.coeff_mat_map = np.argsort(self.coeff_mat.data)
    
        self.coeff_mat_map_A11 = self.coeff_mat_map[self.build_map_coeff_mat(eyeMat_sp, A11_idx)]
        self.coeff_mat_map_A12 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_G_tr, A12_idx, transpose=True)]
        self.coeff_mat_map_A13 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_CqvDq, A13_idx, transpose=True)]
        self.coeff_mat_map_A14 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_Cq, A14_idx, transpose=True)]
        self.coeff_mat_map_A21 = self.coeff_mat_map[self.build_map_coeff_mat(eyeMat_sp, A21_idx)]
        self.coeff_mat_map_M_A22 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_M, A22_idx)]
        self.coeff_mat_map_fv_A22 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_fv, A22_idx, transpose=True)]
        self.coeff_mat_map_A22 = np.unique(np.concatenate([self.coeff_mat_map_M_A22,self.coeff_mat_map_fv_A22]))
        self.coeff_mat_map_A23 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_Cq, A23_idx, transpose=True)]
        self.coeff_mat_map_A32 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_Cq, A32_idx)]
        self.coeff_mat_map_A41 = self.coeff_mat_map[self.build_map_coeff_mat(FreeDyn.MBS_Cq, A41_idx)]
        
        # A21 is set here, as these are const. values
        self.coeff_mat.data[self.coeff_mat_map_A21] = -1.0
        
        return nBDFsys
# -----------------------------------------------------------------------------  

    def idx_temp_init_adj_sys_coeff_mat(self, offset, mtx, transpose=False):
        
        tpl = mtx.T.copy() if transpose else mtx.copy()
        tpl.data = np.arange(offset, offset + tpl.nnz, dtype=int) + 1
        offset += tpl.nnz       
        return offset, tpl
# -----------------------------------------------------------------------------        
    
    def build_map_coeff_mat(self, sub_mat, block_idx_mat, transpose=False):
        
        num_cols = block_idx_mat.shape[1]
    
        target_coo = block_idx_mat.tocoo()
        target_keys = target_coo.row * num_cols + target_coo.col
        sort_idx = np.argsort(target_keys)
        sorted_target_keys = target_keys[sort_idx]
    
        sub_coo = sub_mat.tocoo()
        
        if transpose:
            sub_keys = sub_coo.col * num_cols + sub_coo.row
        else:
            sub_keys = sub_coo.row * num_cols + sub_coo.col
        
        matched_pos = np.searchsorted(sorted_target_keys, sub_keys)
        
        return (block_idx_mat.data[sort_idx[matched_pos]] - 1).astype(np.int32)      
# -----------------------------------------------------------------------------

    def update_coeff_mat_AdjSys(self, FreeDyn, eta0):
        
        """ Update the entries of the coefficient matrix - use directly the non-zeros elements from the API """
        
        # Row 1
        self.coeff_mat.data[self.coeff_mat_map_A11] = eta0 
        self.coeff_mat.data[self.coeff_mat_map_A12] = -FreeDyn.slot_MBS_G_tr.dll_nonzeros
        self.coeff_mat.data[self.coeff_mat_map_A13] = -FreeDyn.slot_MBS_CqvDq.dll_nonzeros
        self.coeff_mat.data[self.coeff_mat_map_A14] = -FreeDyn.slot_MBS_Cq.dll_nonzeros
        
        # Row 2
        # A21 is already set in self.init_adj_sys_coeff_mat
        self.coeff_mat.data[self.coeff_mat_map_A22] = 0.0
        self.coeff_mat.data[self.coeff_mat_map_M_A22] = eta0 * FreeDyn.slot_MBS_M.dll_nonzeros
        self.coeff_mat.data[self.coeff_mat_map_fv_A22] -= FreeDyn.slot_MBS_fv.dll_nonzeros
        self.coeff_mat.data[self.coeff_mat_map_A23] = -FreeDyn.slot_MBS_Cq.dll_nonzeros
    
        # Row 3
        self.coeff_mat.data[self.coeff_mat_map_A32] = FreeDyn.slot_MBS_Cq.dll_nonzeros 
        
        # Row 4
        self.coeff_mat.data[self.coeff_mat_map_A41] = FreeDyn.slot_MBS_Cq.dll_nonzeros 
# -----------------------------------------------------------------------------
    
# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def BDForder1_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT)
        
        # pre-compute sums of matrices
        self.BDF_solVec_J_view_first[:] = - self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        self.BDF_solVec_J_view_second[:] = - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx1, :]

        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF1_eta0, idx2)
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_solVec_Phi_view_first[:] = - self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        self.BDF_solVec_Phi_view_second[:] = - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :]
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF1_eta0, idx2)
        
# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT)
        
        # pre-compute sums of matrices
        self.BDF_solVec_J_view_first[:] = -(self.BDF2_eta1 * self.adjW_J_buff[idx1, :] + self.BDF2_eta2 * self.adjW_J_buff[idx2, :])
        self.BDF_solVec_J_view_second[:] = -(self.BDF2_eta1 * self.BDF_J_buff_M_times_p[idx1, :] + self.BDF2_eta2 * self.BDF_J_buff_M_times_p[idx2, :])

        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF2_eta0, idx2)
# ----------------------------------------------------------------------------- 

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_solVec_Phi_view_first[:] = - (self.BDF2_eta1 * self.adjW_Phi_buff[idx1, :, :] + self.BDF2_eta2 * self.adjW_Phi_buff[idx2, :, :])
        self.BDF_solVec_Phi_view_second[:] = - (self.BDF2_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :] + self.BDF2_eta2 * self.BDF_Phi_buff_M_times_P[idx2, :, :])
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF2_eta0, idx2)

# =============================================================================
# BDF Routine: Shared pre steps 
# =============================================================================

    def BDF_singleStep_J_pre_steps(self, UserFcts, FreeDyn, z):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx1, :] = FreeDyn.MBS_M @ self.adjP_J_buff[idx1, :]
        
        # Updates at BDF time step s_n
        FreeDyn.update_sys_mat_for_BDF()
        UserFcts.get_lagrangian_dq(FreeDyn, z)
        UserFcts.get_lagrangian_dv(FreeDyn, z)
        
        return idx1, idx2
# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_pre_steps(self, FreeDyn):
    
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx1, :, :] = FreeDyn.MBS_M @ self.adjP_Phi_buff[idx1, :, :]
        
        # Updates at BDF time step s_n
        FreeDyn.update_sys_mat_for_BDF()
        
        return idx1, idx2

# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================

    def BDF_singleStep_J_post_steps(self, UserFcts, FreeDyn, eta0, idx2):
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J_view_first[:] += UserFcts.dLdq.T
        self.BDF_solVec_J_view_second[:] += UserFcts.dLdv.T
        
        self.update_coeff_mat_AdjSys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU =  solve(self.BDF_solVec_J)

        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj w and p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:FreeDyn.nDof]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[FreeDyn.nDof:2*FreeDyn.nDof]
# -----------------------------------------------------------------------------

    def BDF_singleStep_Phi_post_steps(self, FreeDyn, eta0, idx2):
        
        self.update_coeff_mat_AdjSys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU = solve(self.BDF_solVec_Phi)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj w and p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:FreeDyn.nDof]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[FreeDyn.nDof:2*FreeDyn.nDof]
# -----------------------------------------------------------------------------