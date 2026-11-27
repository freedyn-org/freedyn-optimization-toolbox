import numpy as np

from .BDF_core import BDF

class DenseBDF(BDF):
    
    def init_memory_layout_dependent(self, FreeDyn, num_xF):
        
        # 1 / eta_0 for order 1,2
        self.BDF1_eta0_inv = 0.0
        self.BDF2_eta0_inv = 0.0
    
        self.dim_block1 = FreeDyn.nDof
        self.dim_block2 = self.dim_block1 + FreeDyn.nConstr

        # coeff. matrix of adjoint system
        dim_sys = self.init_coeff_mat_adj_sys(FreeDyn)
        
        # Note: In post_steps eta_x_adjW_J holds the diff: eta_x_adjW_J - dLdq
        self.eta_x_adjW_J = np.empty(FreeDyn.nDof)
        self.sol_vec_J = np.zeros(dim_sys) 
        if num_xF > 0:
            self.eta_x_adjW_Phi = np.empty((FreeDyn.nDof, num_xF))
            self.sol_vec_Phi = np.zeros((dim_sys, num_xF))
    
# =============================================================================
# BDF System - Matrix layout dense
# =============================================================================

    def init_coeff_mat_adj_sys(self, FreeDyn):
        
       # Allocate the coefficient matrix and create views of the individual blocks
        dim_sys = self.dim_block2 + FreeDyn.nConstr
        self.coeff_mat = np.zeros((dim_sys, dim_sys))
        block1 = slice(None, self.dim_block1)
        block2 = slice(self.dim_block1, self.dim_block2)
        block3 = slice(self.dim_block2, dim_sys)
        self.coeff_mat_view_11 = self.coeff_mat[block1, block1] 
        self.coeff_mat_view_12 = self.coeff_mat[block1, block2]
        self.coeff_mat_view_13 = self.coeff_mat[block1, block3]
        self.coeff_mat_view_21 = self.coeff_mat[block2, block1]
        self.coeff_mat_view_22 = self.coeff_mat[block2, block2]
        self.coeff_mat_view_23 = self.coeff_mat[block2, block3]
        self.coeff_mat_view_31 = self.coeff_mat[block3, block1]
        
        return dim_sys
# -----------------------------------------------------------------------------

    def update_coeff_mat_adj_sys(self, FreeDyn, eta0, eta0_inv):
        
        # Update coefficient matrix entries     
        self.coeff_mat_view_11[:] = eta0 * FreeDyn.SysMat.M - eta0_inv * FreeDyn.SysMat.G - FreeDyn.SysMat.fv_tr
        self.coeff_mat_view_12[:] = -eta0_inv * FreeDyn.SysMat.CqvDq.T - FreeDyn.SysMat.Cq.T
        self.coeff_mat_view_13[:] = -eta0_inv * FreeDyn.SysMat.Cq.T
        self.coeff_mat_view_21[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.G
        self.coeff_mat_view_22[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.CqvDq.T
        self.coeff_mat_view_23[:] = FreeDyn.SysMat.Cq @ FreeDyn.SysMat.Cq.T
        self.coeff_mat_view_31[:] = FreeDyn.SysMat.Cq

# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def order1_post_steps_J(self, UserFcts, FreeDyn):
        self.BDF1_eta0_inv = 1 / self.BDF1_eta0
        
        # Compute linear combinations
        np.multiply(self.BDF1_eta1, self.buff_adjW_J[self.buff_idx1, :], out = self.eta_x_adjW_J)
        self.sol_vec_J[:self.dim_block1] = - self.BDF1_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :]

        self.shared_post_steps_J(UserFcts, FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)
        
    def order1_post_steps_Phi(self, FreeDyn):
        self.BDF1_eta0_inv = 1 / self.BDF1_eta0
        
        # Compute linear combinations
        np.multiply(self.BDF1_eta1, self.buff_adjW_Phi[self.buff_idx1, :, :], out = self.eta_x_adjW_Phi)
        self.sol_vec_Phi[:self.dim_block1, :] = - self.BDF1_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :]

        self.shared_post_steps_Phi(FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)
        
# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def order2_post_steps_J(self, UserFcts, FreeDyn):

        self.BDF2_eta0_inv = 1 / self.BDF2_eta0
        
        # Compute linear combinations
        np.multiply(self.BDF2_eta1, self.buff_adjW_J[self.buff_idx1, :], out = self.eta_x_adjW_J)
        self.eta_x_adjW_J += self.BDF2_eta2 * self.buff_adjW_J[self.buff_idx2, :]
        self.sol_vec_J[:self.dim_block1] = - self.BDF2_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :] 
        self.sol_vec_J[:self.dim_block1] -= self.BDF2_eta2 * self.buff_M_x_AdjP_J[self.buff_idx2, :]

        self.shared_post_steps_J(UserFcts, FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv)
        
    def order2_post_steps_Phi(self, FreeDyn):

        self.BDF2_eta0_inv = 1 / self.BDF2_eta0
        
        # Compute linear combinations
        np.multiply(self.BDF2_eta1, self.buff_adjW_Phi[self.buff_idx1, :, :], out = self.eta_x_adjW_Phi)
        self.eta_x_adjW_Phi += self.BDF2_eta2 * self.buff_adjW_Phi[self.buff_idx2, :, :]
        self.sol_vec_Phi[:self.dim_block1, :] = - self.BDF2_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] 
        self.sol_vec_Phi[:self.dim_block1, :] -= self.BDF2_eta2 * self.buff_M_x_AdjP_Phi[self.buff_idx2, :, :]

        self.shared_post_steps_Phi(FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv)
        
# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================
    def shared_post_steps_J(self, UserFcts, FreeDyn, eta0, eta0_inv):
        
        # Update system matricies at BDF step s_n
        FreeDyn.update_sys_mat_for_BDF()
        
        # Modify eta_x_adjW_J for pre-computing the diff
        self.eta_x_adjW_J -= UserFcts.dLdq.T
        
        # Update coeff. matrix and add missing terms to the solution vector
        self.update_coeff_mat_adj_sys(FreeDyn, eta0, eta0_inv)
        self.sol_vec_J[:self.dim_block1] += UserFcts.dLdv.T
        self.sol_vec_J[:self.dim_block1] -= eta0_inv * self.eta_x_adjW_J
        self.sol_vec_J[self.dim_block1:self.dim_block2] = FreeDyn.SysMat.Cq @ self.eta_x_adjW_J
        
        # Compute adj_vars p, sigma, mu at BDF step s_n
        vec_P_Sig_MU = np.linalg.solve(self.coeff_mat, self.sol_vec_J)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj p at time idx = 0
        self.adjP_J[:] = vec_P_Sig_MU[:self.dim_block1]
        
        # Compute adj w at time idx = 0
        self.buff_adjW_J[self.buff_idx1, :] = FreeDyn.SysMat.G @ self.adjP_J
        self.buff_adjW_J[self.buff_idx1, :] += FreeDyn.SysMat.CqvDq.T @ vec_P_Sig_MU[self.dim_block1:self.dim_block2] 
        self.buff_adjW_J[self.buff_idx1, :] += FreeDyn.SysMat.Cq.T @ vec_P_Sig_MU[self.dim_block2:] 
        self.buff_adjW_J[self.buff_idx1, :] -= self.eta_x_adjW_J
        self.buff_adjW_J[self.buff_idx1, :] *= eta0_inv
# ----------------------------------------------------------------------------- 

    def shared_post_steps_Phi(self, FreeDyn, eta0, eta0_inv):
        
        # Update system matricies at BDF step s_n
        FreeDyn.update_sys_mat_for_BDF()
        
        # Update coeff. matrix and add missing terms to the solution vector
        self.update_coeff_mat_adj_sys(FreeDyn, eta0, eta0_inv)
        self.sol_vec_Phi[:self.dim_block1, :] -= eta0_inv * self.eta_x_adjW_Phi
        self.sol_vec_Phi[self.dim_block1:self.dim_block2, :] = FreeDyn.SysMat.Cq @ self.eta_x_adjW_Phi
        
        # Compute adj_vars p, sigma, mu at BDF step s_n
        vec_P_Sig_MU = np.linalg.solve(self.coeff_mat, self.sol_vec_Phi) 
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj p at time idx = 0
        self.adjP_Phi[:] = vec_P_Sig_MU[:self.dim_block1]
        
        # Compute adj w at time idx = 0
        self.buff_adjW_Phi[self.buff_idx1, :, :] = FreeDyn.SysMat.G @ self.adjP_Phi
        self.buff_adjW_Phi[self.buff_idx1, :, :] += FreeDyn.SysMat.CqvDq.T @ vec_P_Sig_MU[self.dim_block1:self.dim_block2,:] 
        self.buff_adjW_Phi[self.buff_idx1, :, :] += FreeDyn.SysMat.Cq.T @ vec_P_Sig_MU[self.dim_block2:,:] 
        self.buff_adjW_Phi[self.buff_idx1, :, :] -= self.eta_x_adjW_Phi
        self.buff_adjW_Phi[self.buff_idx1, :, :] *= eta0_inv