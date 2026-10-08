import numpy as np

from core.BDF_core import CoreBDF

class DenseBDF(CoreBDF):
    
    def __init__(self, dataOpt, FreeDyn):
        nBDFsys = self.init_coeffMat_AdjSys(FreeDyn)
        partition_solVec_1 = FreeDyn.nDof
        partition_solVec_2 = FreeDyn.nDofConstr
        CoreBDF.__init__(self, dataOpt, FreeDyn, nBDFsys, partition_solVec_1, partition_solVec_2)
        CoreBDF.allocate_buffs_BDF_add_dense(self, FreeDyn.nDof, dataOpt.num_xF)        
        return None
    
# =============================================================================
# BDF System - Matrix layout dense
# =============================================================================

    def init_coeffMat_AdjSys(self, FreeDyn):
        
        """ Allocate the coefficient matrix and create views of the individual blocks """
        
        nBDFsys = FreeDyn.nDof + 2*FreeDyn.nConstr
        self.BDF_coeffMat = np.zeros((nBDFsys, nBDFsys))
        
        block1 = slice(None, FreeDyn.nDof)
        block2 = slice(FreeDyn.nDof, FreeDyn.nDofConstr)
        block3 = slice(FreeDyn.nDofConstr, nBDFsys)
    
        self.BDF_coeffMat_view_11 = self.BDF_coeffMat[block1, block1] 
        self.BDF_coeffMat_view_12 = self.BDF_coeffMat[block1, block2]
        self.BDF_coeffMat_view_13 = self.BDF_coeffMat[block1, block3]
        self.BDF_coeffMat_view_21 = self.BDF_coeffMat[block2, block1]
        self.BDF_coeffMat_view_22 = self.BDF_coeffMat[block2, block2]
        self.BDF_coeffMat_view_23 = self.BDF_coeffMat[block2, block3]
        self.BDF_coeffMat_view_31 = self.BDF_coeffMat[block3, block1]
        
        return nBDFsys
# -----------------------------------------------------------------------------

    def update_coeffMat_AdjSys(self, FreeDyn, eta0, eta0_inv):
        
        """ Update the entries of the coefficient matrix - use the dense matricies """
        
        self.BDF_coeffMat_view_11[:] = eta0 * FreeDyn.MBS_M - eta0_inv * FreeDyn.MBS_G_tr.T - FreeDyn.MBS_fv.T
        self.BDF_coeffMat_view_12[:] = -eta0_inv * FreeDyn.MBS_CqvDq.T - FreeDyn.MBS_Cq.T
        self.BDF_coeffMat_view_13[:] = -eta0_inv * FreeDyn.MBS_Cq.T
        self.BDF_coeffMat_view_21[:] = FreeDyn.MBS_Cq @ FreeDyn.MBS_G_tr.T
        self.BDF_coeffMat_view_22[:] = FreeDyn.MBS_Cq @ FreeDyn.MBS_CqvDq.T
        self.BDF_coeffMat_view_23[:] = FreeDyn.MBS_Cq @ FreeDyn.MBS_Cq.T
        self.BDF_coeffMat_view_31[:] = FreeDyn.MBS_Cq
# -----------------------------------------------------------------------------

# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================

    def BDForder1_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_J_eta_times_adjW = self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        self.BDF_solVec_J_view_first[:] = - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx1, :]

        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv, idx2)
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder1_coeffs_eta(idx1, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_Phi_eta_times_adjW = self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        self.BDF_solVec_Phi_view_first[:] = - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :]

        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv, idx2)

# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_J_pre_steps(UserFcts, FreeDyn, z)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT) 
        
        # pre-compute sums of matrices
        self.BDF_J_eta_times_adjW = self.BDF2_eta1 * self.adjW_J_buff[idx1, :] + self.BDF2_eta2 * self.adjW_J_buff[idx2, :]
        self.BDF_solVec_J_view_first[:] = - (self.BDF2_eta1 * self.BDF_J_buff_M_times_p[idx1, :] + self.BDF2_eta2 * self.BDF_J_buff_M_times_p[idx2, :])
        
        
        self.BDF_singleStep_J_post_steps(UserFcts, FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv, idx2)
# -----------------------------------------------------------------------------  

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        
        idx1, idx2 = self.BDF_singleStep_Phi_pre_steps(FreeDyn)
        
        # Compute BDF coefficients
        self.get_BDForder2_coeffs_eta(idx1, idx2, deltaT)
        
        # pre-compute sums of matrices
        self.BDF_Phi_eta_times_adjW = self.BDF2_eta1 * self.adjW_Phi_buff[idx1, :, :] + self.BDF2_eta2 * self.adjW_Phi_buff[idx2, :, :]
        self.BDF_solVec_Phi_view_first[:] = - (self.BDF2_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :] + self.BDF2_eta2 * self.BDF_Phi_buff_M_times_P[idx2, :, :])

        
        self.BDF_singleStep_Phi_post_steps(FreeDyn, self.BDF2_eta0, self.BDF2_eta0_inv, idx2)     

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
    def BDF_singleStep_J_post_steps(self, UserFcts, FreeDyn, eta0, eta0_inv, idx2):
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J_view_first[:] += UserFcts.dLdv.T + eta0_inv*(UserFcts.dLdq.T - self.BDF_J_eta_times_adjW)
        self.BDF_solVec_J_view_second[:] = FreeDyn.MBS_Cq @ (self.BDF_J_eta_times_adjW - UserFcts.dLdq.T)
        
        self.update_coeffMat_AdjSys(FreeDyn, eta0, eta0_inv)
        vec_P_Sig_MU = np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_J)
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj p at time idx = 0
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_P_Sig_MU[:FreeDyn.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = FreeDyn.MBS_G_tr.T @ vec_P_Sig_MU[:FreeDyn.nDof]
        self.adjW_J_buff[self.BDF_idx_buff, :] += FreeDyn.MBS_CqvDq.T @ vec_P_Sig_MU[FreeDyn.nDof:FreeDyn.nDofConstr] 
        self.adjW_J_buff[self.BDF_idx_buff, :] += FreeDyn.MBS_Cq.T @ vec_P_Sig_MU[FreeDyn.nDofConstr:] 
        self.adjW_J_buff[self.BDF_idx_buff, :] -= self.BDF_J_eta_times_adjW
        self.adjW_J_buff[self.BDF_idx_buff, :] += UserFcts.dLdq.T 
        self.adjW_J_buff[self.BDF_idx_buff, :] *= eta0_inv
# ----------------------------------------------------------------------------- 

    def BDF_singleStep_Phi_post_steps(self, FreeDyn, eta0, eta0_inv, idx2):
        
        # Bulid the solution vector of the Adjoint Sys      
        self.BDF_solVec_Phi_view_first[:] -= eta0_inv * self.BDF_Phi_eta_times_adjW
        self.BDF_solVec_Phi_view_second[:] = FreeDyn.MBS_Cq @ self.BDF_Phi_eta_times_adjW
        
        self.update_coeffMat_AdjSys(FreeDyn, eta0, eta0_inv)
        vec_P_Sig_MU = np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_Phi) 
    
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj p at time idx = 0
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_P_Sig_MU[:FreeDyn.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = FreeDyn.MBS_G_tr.T @ vec_P_Sig_MU[:FreeDyn.nDof]
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += FreeDyn.MBS_CqvDq.T @ vec_P_Sig_MU[FreeDyn.nDof:FreeDyn.nDofConstr] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += FreeDyn.MBS_Cq.T @ vec_P_Sig_MU[FreeDyn.nDofConstr:] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= self.BDF_Phi_eta_times_adjW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= eta0_inv
  