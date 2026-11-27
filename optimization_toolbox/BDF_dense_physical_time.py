import numpy as np

from .BDF_core import BDF

class DenseBDF(BDF):
    
    def init_memory_layout_dependent(self, FreeDyn, num_xF):
    
        self.dim_block1 = FreeDyn.nDof
        self.dim_block2 = self.dim_block1 + FreeDyn.nConstr
        
        # Note: In post_steps buff_eta_x_adjW_J holds the diff: buff_eta_x_adjW_J - dLdq
        self.buff_eta_x_adjW_J = np.empty(FreeDyn.nDof)
        if num_xF > 0:
            self.buff_eta_x_adjW_Phi = np.empty((FreeDyn.nDof, num_xF)) 
    
# =============================================================================
# BDF System - Matrix layout dense
# =============================================================================

    def init_coeff_mat_adj_sys(self, FreeDyn):
        
        """ Allocate the coefficient matrix and create views of the individual blocks """
        
        nBDFsys = self.dim_block2 + FreeDyn.nConstr
        block1 = slice(None, self.dim_block1)
        block2 = slice(self.dim_block1, self.dim_block2)
        block3 = slice(self.dim_block2, nBDFsys)
        
        self.BDF_coeffMat = np.zeros((nBDFsys, nBDFsys))
        self.BDF_coeffMat_view_11 = self.BDF_coeffMat[block1, block1] 
        self.BDF_coeffMat_view_12 = self.BDF_coeffMat[block1, block2]
        self.BDF_coeffMat_view_13 = self.BDF_coeffMat[block1, block3]
        self.BDF_coeffMat_view_21 = self.BDF_coeffMat[block2, block1]
        self.BDF_coeffMat_view_22 = self.BDF_coeffMat[block2, block2]
        self.BDF_coeffMat_view_23 = self.BDF_coeffMat[block2, block3]
        self.BDF_coeffMat_view_31 = self.BDF_coeffMat[block3, block1]
        
        return nBDFsys
# -----------------------------------------------------------------------------

    def update_coeff_mat_adj_sys(self, FreeDyn, eta0, eta0_inv):
        
        """ Update the entries of the coefficient matrix - use the dense matricies """
        
        self.BDF_coeffMat_view_11[:] = eta0 * FreeDyn.SysMat.M - eta0_inv * FreeDyn.SysMat.G - FreeDyn.SysMat.fv_tr
        self.BDF_coeffMat_view_12[:] = -eta0_inv * FreeDyn.SysMat.CqvDq.T - FreeDyn.SysMat.Cq.T
        self.BDF_coeffMat_view_13[:] = -eta0_inv * FreeDyn.SysMat.Cq.T
        self.BDF_coeffMat_view_21[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.G
        self.BDF_coeffMat_view_22[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.CqvDq.T
        self.BDF_coeffMat_view_23[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.Cq.T
        self.BDF_coeffMat_view_31[:] = FreeDyn.SysMat.Cq
# -----------------------------------------------------------------------------

# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def BDForder1_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(deltaT) 
        
        # pre-compute sums of matrices
        self.buff_eta_x_adjW_J = self.BDF1_eta1 * self.buff_adjW_J[self.buff_idx1, :]
        self.buff_solVec_J[:self.dim_block1] = - self.BDF1_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :]

        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        
        self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(deltaT) 
        
        # pre-compute sums of matrices
        self.buff_eta_x_adjW_Phi = self.BDF1_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :]
        self.buff_solVec_Phi[:self.dim_block1, :] = - self.BDF1_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :]

        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)

# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(deltaT) 
        
        # pre-compute sums of matrices
        self.buff_eta_x_adjW_J = self.BDF2_eta1 * self.buff_adjW_J[self.buff_idx1, :] 
        self.buff_eta_x_adjW_J += self.BDF2_eta2 * self.buff_adjW_J[self.buff_idx2, :]
        self.buff_solVec_J[:self.dim_block1] = - self.BDF2_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :] 
        self.buff_solVec_J[:self.dim_block1] -= self.BDF2_eta2 * self.buff_M_x_AdjP_J[self.buff_idx2, :]
        
        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv)
# -----------------------------------------------------------------------------  

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        
        self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(deltaT)
        
        # pre-compute sums of matrices
        self.buff_eta_x_adjW_Phi = self.BDF2_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :] 
        self.buff_eta_x_adjW_Phi += self.BDF2_eta2 * self.buff_adjW_Phi[self.buff_idx2, :, :]
        self.buff_solVec_Phi[:self.dim_block1, :] = - self.BDF2_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] 
        self.buff_solVec_Phi[:self.dim_block1, :] -= self.BDF2_eta2 * self.buff_M_x_AdjP_Phi[self.buff_idx2, :, :]

        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv)     

# =============================================================================
# BDF Routine: Shared pre steps 
# =============================================================================

    def BDF_singleStep_J_pre_steps(self, UserFcts, FreeDyn, z):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_J[self.buff_idx1, :] = FreeDyn.SysMat.M @ self.buff_adjP_J[self.buff_idx1, :]

        # Updates at BDF time step s_n
        FreeDyn.update_sys_mat_for_BDF()
        UserFcts.get_lagrangian_dq(FreeDyn, z)
        UserFcts.get_lagrangian_dv(FreeDyn, z)
# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_pre_steps(self, FreeDyn):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] = FreeDyn.SysMat.M @ self.buff_adjP_Phi[self.buff_idx1, :, :]
        
        # Updates at BDF time step s_n
        FreeDyn.update_sys_mat_for_BDF()

# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================
    def BDF_singleStep_J_post_steps(self, UserFcts, FreeDyn, eta0, eta0_inv):
        
        # Modify buff_eta_x_adjW_J for pre-computing the diff
        self.buff_eta_x_adjW_J -= UserFcts.dLdq.T
        
        # Bulid the solution vector of the Adjoint Sys
        self.buff_solVec_J[:self.dim_block1] += UserFcts.dLdv.T
        self.buff_solVec_J[:self.dim_block1] -= eta0_inv * self.buff_eta_x_adjW_J
        self.buff_solVec_J[self.dim_block1:self.dim_block2] = FreeDyn.SysMat.Cq @ self.buff_eta_x_adjW_J
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0, eta0_inv)
        vec_P_Sig_MU = np.linalg.solve(self.BDF_coeffMat, self.buff_solVec_J)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj p at time idx = 0
        self.buff_adjP_J[self.buff_idx1, :] = vec_P_Sig_MU[:self.dim_block1]
        
        # Compute adj w at time idx = 0
        self.buff_adjW_J[self.buff_idx1, :] = FreeDyn.SysMat.G @ vec_P_Sig_MU[:self.dim_block1]
        self.buff_adjW_J[self.buff_idx1, :] += FreeDyn.SysMat.CqvDq.T @ vec_P_Sig_MU[self.dim_block1:self.dim_block2] 
        self.buff_adjW_J[self.buff_idx1, :] += FreeDyn.SysMat.Cq.T @ vec_P_Sig_MU[self.dim_block2:] 
        self.buff_adjW_J[self.buff_idx1, :] -= self.buff_eta_x_adjW_J
        self.buff_adjW_J[self.buff_idx1, :] *= eta0_inv
# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_post_steps(self, FreeDyn, eta0, eta0_inv):
        
        # Bulid the solution vector of the Adjoint Sys      
        self.buff_solVec_Phi[:self.dim_block1, :] -= eta0_inv * self.buff_eta_x_adjW_Phi
        self.buff_solVec_Phi[self.dim_block1:self.dim_block2, :] = FreeDyn.SysMat.Cq @ self.buff_eta_x_adjW_Phi
        
        self.update_coeff_mat_adj_sys(FreeDyn, eta0, eta0_inv)
        vec_P_Sig_MU = np.linalg.solve(self.BDF_coeffMat, self.buff_solVec_Phi) 
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj p at time idx = 0
        self.buff_adjP_Phi[self.buff_idx1, :, :] = vec_P_Sig_MU[:self.dim_block1]
        
        # Compute adj w at time idx = 0
        self.buff_adjW_Phi[self.buff_idx1, :, :] = FreeDyn.SysMat.G @ vec_P_Sig_MU[:self.dim_block1,:]
        self.buff_adjW_Phi[self.buff_idx1, :, :] += FreeDyn.SysMat.CqvDq.T @ vec_P_Sig_MU[self.dim_block1:self.dim_block2,:] 
        self.buff_adjW_Phi[self.buff_idx1, :, :] += FreeDyn.SysMat.Cq.T @ vec_P_Sig_MU[self.dim_block2:,:] 
        self.buff_adjW_Phi[self.buff_idx1, :, :] -= self.buff_eta_x_adjW_Phi
        self.buff_adjW_Phi[self.buff_idx1, :, :] *= eta0_inv
  