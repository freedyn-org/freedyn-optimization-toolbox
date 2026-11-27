import numpy as np
from scipy.sparse.linalg import factorized

from .BDF_core import BDF

class SparseBDF(BDF):
    
    def init_memory_layout_dependent(self, FreeDyn, num_xF):

        self.dim_block1 = FreeDyn.nDof
        self.dim_block2 = self.dim_block1 + FreeDyn.nDof
    
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
        
        self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(deltaT)
        
        # pre-compute sums of matrices
        self.buff_solVec_J[:self.dim_block1] = - self.BDF1_eta1 * self.buff_adjW_J[self.buff_idx1, :]
        self.buff_solVec_J[self.dim_block1:self.dim_block2] = - self.BDF1_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :]

        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF1_eta0)
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        
        self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(deltaT) 
        
        # pre-compute sums of matrices
        self.buff_solVec_Phi[:self.dim_block1,:] = - self.BDF1_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :]
        self.buff_solVec_Phi[self.dim_block1:self.dim_block2,:] = - self.BDF1_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :]
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF1_eta0)
        
# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(deltaT)
        
        # pre-compute sums of matrices
        self.buff_solVec_J[:self.dim_block1] = - self.BDF2_eta1 * self.buff_adjW_J[self.buff_idx1, :] 
        self.buff_solVec_J[:self.dim_block1] -= self.BDF2_eta2 * self.buff_adjW_J[self.buff_idx2, :]
        self.buff_solVec_J[self.dim_block1:self.dim_block2] = - self.BDF2_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :] 
        self.buff_solVec_J[self.dim_block1:self.dim_block2] -= self.BDF2_eta2 * self.buff_M_x_AdjP_J[self.buff_idx2, :]

        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF2_eta0)
# ----------------------------------------------------------------------------- 

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        
        self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(deltaT) 
        
        # pre-compute sums of matrices
        self.buff_solVec_Phi[:self.dim_block1, :] = - self.BDF2_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :]
        self.buff_solVec_Phi[:self.dim_block1, :] -= self.BDF2_eta2 * self.buff_adjW_Phi[self.buff_idx2, :, :]
        self.buff_solVec_Phi[self.dim_block1:self.dim_block2,:] = - self.BDF2_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] 
        self.buff_solVec_Phi[self.dim_block1:self.dim_block2,:] -= self.BDF2_eta2 * self.buff_M_x_AdjP_Phi[self.buff_idx2, :, :]
        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF2_eta0)

# =============================================================================
# BDF Routine: Shared pre steps 
# =============================================================================

    def BDF_singleStep_J_pre_steps(self, UserFcts, FreeDyn, z):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_J[self.buff_idx1, :] = FreeDyn.SysMat.M @ self.buff_adjP_J[self.buff_idx1, :]
        
        # Updates at BDF time step s_n
        UserFcts.get_lagrangian_dq(FreeDyn, z)
        UserFcts.get_lagrangian_dv(FreeDyn, z)
        
        # Sys Matrices are updated within the update of the coeff matrix

# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_pre_steps(self, FreeDyn):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] = FreeDyn.SysMat.M @ self.buff_adjP_Phi[self.buff_idx1, :, :]
        
        # Sys Matrices are updated within the update of the coeff matrix

# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================

    def BDF_singleStep_J_post_steps(self, UserFcts, FreeDyn, eta0):
        
        # Bulid the solution vector of the Adjoint Sys
        self.buff_solVec_J[:self.dim_block1] += UserFcts.dLdq.T
        self.buff_solVec_J[self.dim_block1:self.dim_block2] += UserFcts.dLdv.T
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU =  solve(self.buff_solVec_J)

        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj w and p at time idx = 0
        self.buff_adjW_J[self.buff_idx1, :] = vec_W_P_Sig_MU[:self.dim_block1]
        self.buff_adjP_J[self.buff_idx1, :] = vec_W_P_Sig_MU[self.dim_block1:self.dim_block2]
# -----------------------------------------------------------------------------

    def BDF_singleStep_Phi_post_steps(self, FreeDyn, eta0):
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU = solve(self.buff_solVec_Phi)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj w and p at time idx = 0
        self.buff_adjW_Phi[self.buff_idx1, :, :] = vec_W_P_Sig_MU[:self.dim_block1,:]
        self.buff_adjP_Phi[self.buff_idx1, :, :] = vec_W_P_Sig_MU[self.dim_block1:self.dim_block2,:]
# -----------------------------------------------------------------------------