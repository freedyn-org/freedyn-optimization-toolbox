import numpy as np
from scipy.sparse.linalg import factorized

from .BDF_core import CoreBDF


class SparseBDF(CoreBDF):
    
# =============================================================================
# BDF System - Matrix layout sparse
# =============================================================================

    def init_coeff_mat_adj_sys(self, FreeDyn):
        
        """ Allocate the coefficient matrix and create maps of the individual blocks """
        nBDFsys = 2 * (FreeDyn.nDof + FreeDyn.nConstr)
        
        FreeDyn.API.compute_initial_conditions()
        mat = FreeDyn.SysMat.BDF_mat.copy()
        mat.data = np.arange(mat.nnz, dtype=int)
        idx_csc = mat.tocsc()
        self.coeff_mat = idx_csc.astype(np.float64)
        self.coeff_mat_map = np.empty(mat.nnz, dtype=int)
        self.coeff_mat_map[idx_csc.data] = np.arange(mat.nnz, dtype=int)
        
        return nBDFsys     
# -----------------------------------------------------------------------------

    def update_coeff_mat_adj_sys(self, FreeDyn, eta0):
        
        """ Update the entries of the coefficient matrix - use directly the non-zeros elements from the API """
        FreeDyn.SysMat.update_BDF_mat_eta(eta0)
        FreeDyn.update_sys_mat_for_BDF()
        self.coeff_mat.data[self.coeff_mat_map] = FreeDyn.SysMat.slot_BDF_mat.dll_nonzeros
    
# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def BDForder1_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT)
        
        # pre-compute sums of matrices
        self.BDF_solVec_J[:self.view_block1] = - self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        self.BDF_solVec_J[self.view_block1:self.view_block2] = - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx1, :]

        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF1_eta0, idx2)
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_solVec_Phi[:self.view_block1,:] = - self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        self.BDF_solVec_Phi[self.view_block1:self.view_block2,:] = - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :]
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF1_eta0, idx2)
        
# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT)
        
        # pre-compute sums of matrices
        self.BDF_solVec_J[:self.view_block1] = -(self.BDF2_eta1 * self.adjW_J_buff[idx1, :] + self.BDF2_eta2 * self.adjW_J_buff[idx2, :])
        self.BDF_solVec_J[self.view_block1:self.view_block2] = -(self.BDF2_eta1 * self.BDF_J_buff_M_times_p[idx1, :] + self.BDF2_eta2 * self.BDF_J_buff_M_times_p[idx2, :])

        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF2_eta0, idx2)
# ----------------------------------------------------------------------------- 

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_solVec_Phi[:self.view_block1, :] = - (self.BDF2_eta1 * self.adjW_Phi_buff[idx1, :, :] + self.BDF2_eta2 * self.adjW_Phi_buff[idx2, :, :])
        self.BDF_solVec_Phi[self.view_block1:self.view_block2,:] = - (self.BDF2_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :] + self.BDF2_eta2 * self.BDF_Phi_buff_M_times_P[idx2, :, :])
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF2_eta0, idx2)

# =============================================================================
# BDF Routine: Shared pre steps 
# =============================================================================

    def BDF_singleStep_J_pre_steps(self, UserFcts, FreeDyn, z):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx1, :] = FreeDyn.SysMat.M @ self.adjP_J_buff[idx1, :]
        
        # Updates at BDF time step s_n
        UserFcts.get_lagrangian_dq(FreeDyn, z)
        UserFcts.get_lagrangian_dv(FreeDyn, z)
        
        # Sys Matrices are updated within the update of the coeff matrix
        
        return idx1, idx2
# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_pre_steps(self, FreeDyn):
    
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx1, :, :] = FreeDyn.SysMat.M @ self.adjP_Phi_buff[idx1, :, :]
        
        # Sys Matrices are updated within the update of the coeff matrix
        
        return idx1, idx2

# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================

    def BDF_singleStep_J_post_steps(self, UserFcts, FreeDyn, eta0, idx2):
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.view_block1] += UserFcts.dLdq.T
        self.BDF_solVec_J[self.view_block1:self.view_block2] += UserFcts.dLdv.T
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU =  solve(self.BDF_solVec_J)

        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj w and p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:self.view_block1]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[self.view_block1:self.view_block2]
# -----------------------------------------------------------------------------

    def BDF_singleStep_Phi_post_steps(self, FreeDyn, eta0, idx2):
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU = solve(self.BDF_solVec_Phi)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj w and p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:self.view_block1,:]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[self.view_block1:self.view_block2,:]
# -----------------------------------------------------------------------------