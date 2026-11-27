import numpy as np
from scipy.sparse.linalg import factorized

from .BDF_core import BDF

class SparseBDF(BDF):
    
    def init_memory_layout_dependent(self, FreeDyn, num_xF):

        self.dim_block1 = FreeDyn.nDof
        self.dim_block2 = self.dim_block1 + FreeDyn.nDof
        dim_sys = 2 * (FreeDyn.nDof + FreeDyn.nConstr)
        self.init_coeff_mat_adj_sys(FreeDyn)
        
        self.sol_vec_J = np.zeros(dim_sys)
        if num_xF > 0:
            self.sol_vec_Phi = np.zeros((dim_sys, num_xF))
    
# =============================================================================
# BDF System - Matrix layout sparse
# =============================================================================

    def init_coeff_mat_adj_sys(self, FreeDyn):
        
        # Allocate the coefficient matrix and create maps of the individual blocks
        FreeDyn.API.compute_initial_conditions()
        mat = FreeDyn.SysMat.BDF_mat.copy()
        mat.data = np.arange(mat.nnz, dtype=int)
        idx_csc = mat.tocsc()
        self.coeff_mat = idx_csc.astype(np.float64)
        self.coeff_mat_map = np.empty(mat.nnz, dtype=int)
        self.coeff_mat_map[idx_csc.data] = np.arange(mat.nnz, dtype=int) 
# -----------------------------------------------------------------------------

    def update_coeff_mat_adj_sys(self, FreeDyn, eta0):
        
        # Update coefficient matrix entries - use directly the non-zeros elements
        FreeDyn.SysMat.update_BDF_mat_eta(eta0)
        FreeDyn.update_sys_mat_for_BDF()
        self.coeff_mat.data[self.coeff_mat_map] = FreeDyn.SysMat.slot_BDF_mat.dll_nonzeros
    
# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def order1_post_steps_J(self, UserFcts, FreeDyn):
        
        # Compute linear combinations
        self.sol_vec_J[:self.dim_block1] = - self.BDF1_eta1 * self.buff_adjW_J[self.buff_idx1, :]
        self.sol_vec_J[self.dim_block1:self.dim_block2] = - self.BDF1_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :]

        self.shared_post_steps_J(UserFcts, FreeDyn, self.BDF1_eta0)

    def order1_post_steps_Phi(self, FreeDyn):
        
        # Compute linear combinations
        self.sol_vec_Phi[:self.dim_block1,:] = - self.BDF1_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :]
        self.sol_vec_Phi[self.dim_block1:self.dim_block2,:] = - self.BDF1_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :]

        self.shared_post_steps_Phi(FreeDyn, self.BDF1_eta0)
        
# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def order2_post_steps_J(self, UserFcts, FreeDyn):

        # Compute linear combinations
        self.sol_vec_J[:self.dim_block1] = - self.BDF2_eta1 * self.buff_adjW_J[self.buff_idx1, :] 
        self.sol_vec_J[:self.dim_block1] -= self.BDF2_eta2 * self.buff_adjW_J[self.buff_idx2, :]
        self.sol_vec_J[self.dim_block1:self.dim_block2] = - self.BDF2_eta1 * self.buff_M_x_AdjP_J[self.buff_idx1, :] 
        self.sol_vec_J[self.dim_block1:self.dim_block2] -= self.BDF2_eta2 * self.buff_M_x_AdjP_J[self.buff_idx2, :]

        self.shared_post_steps_J(UserFcts, FreeDyn, self.BDF2_eta0)

    def order2_post_steps_Phi(self, FreeDyn):

        # Compute linear combinations
        self.sol_vec_Phi[:self.dim_block1, :] = - self.BDF2_eta1 * self.buff_adjW_Phi[self.buff_idx1, :, :]
        self.sol_vec_Phi[:self.dim_block1, :] -= self.BDF2_eta2 * self.buff_adjW_Phi[self.buff_idx2, :, :]
        self.sol_vec_Phi[self.dim_block1:self.dim_block2,:] = - self.BDF2_eta1 * self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] 
        self.sol_vec_Phi[self.dim_block1:self.dim_block2,:] -= self.BDF2_eta2 * self.buff_M_x_AdjP_Phi[self.buff_idx2, :, :]

        self.shared_post_steps_Phi(FreeDyn, self.BDF2_eta0)

# =============================================================================
# BDF Routine: Shared post steps 
# =============================================================================

    def shared_post_steps_J(self, UserFcts, FreeDyn, eta0):
        
        # Update system matricies at BDF step s_n within update of BDF coeff. matrix
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        
        # Add missing terms to the solution vector
        self.sol_vec_J[:self.dim_block1] += UserFcts.dLdq.T
        self.sol_vec_J[self.dim_block1:self.dim_block2] += UserFcts.dLdv.T
        
        # Compute adj_vars p, sigma, mu at BDF step s_n
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU = solve(self.sol_vec_J)

        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj w and p at time idx = 0
        self.buff_adjW_J[self.buff_idx1, :] = vec_W_P_Sig_MU[:self.dim_block1]
        self.adjP_J[:] = vec_W_P_Sig_MU[self.dim_block1:self.dim_block2]
# -----------------------------------------------------------------------------

    def shared_post_steps_Phi(self, FreeDyn, eta0):
        
        # Update system matricies at BDF step s_n within update of BDF coeff. matrix
        self.update_coeff_mat_adj_sys(FreeDyn, eta0)
        
        # Compute adj_vars p, sigma, mu at BDF step s_n
        solve = factorized(self.coeff_mat)
        vec_W_P_Sig_MU = solve(self.sol_vec_Phi)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.shift_buff_indices()
        
        # Compute adj w and p at time idx = 0
        self.buff_adjW_Phi[self.buff_idx1, :, :] = vec_W_P_Sig_MU[:self.dim_block1,:]
        self.adjP_Phi[:] = vec_W_P_Sig_MU[self.dim_block1:self.dim_block2,:]
# -----------------------------------------------------------------------------