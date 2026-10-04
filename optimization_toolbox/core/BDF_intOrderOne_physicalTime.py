import core.BDF_coeffMat as coeffMat

class BDF_intOrderOne:
    
    def __init__(self, MBS_modeMAT_sparse):
        
        self.BDF1_eta0_inv = 0.0
        self.BDF1_eta0 = 0.0
        self.BDF1_eta1 = 0.0
        
        if MBS_modeMAT_sparse:    
            self.BDForder1_singleStep_J = self.BDForder1_singleStep_J_sparse
            self.BDForder1_singleStep_Phi = self.BDForder1_singleStep_Phi_sparse
        else:
            self.BDForder1_singleStep_J = self.BDForder1_singleStep_J_dense
            self.BDForder1_singleStep_Phi = self.BDForder1_singleStep_Phi_dense   
        
        print('BDF integration order 1 initialized')   
# -----------------------------------------------------------------------------

    def get_BDForder1_coeffs_eta(self, idx):

        self.BDF1_eta0_inv = self.BDF_diff_tau[idx]
        self.BDF1_eta0 = 1 / self.BDF1_eta0_inv
        self.BDF1_eta1 = -self.BDF1_eta0 
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_J_dense(self, user_fcts, FreeDyn, z, deltaT):
        
        idx = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx, :] = FreeDyn.MBS_M @ self.adjP_J_buff[idx, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat(FreeDyn)
        user_fcts.get_Lagrangian_dq(FreeDyn, z)
        user_fcts.get_Lagrangian_dv(FreeDyn, z)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx] = deltaT
        self.get_BDForder1_coeffs_eta(idx) 
        
        # pre-compute sums of matrices
        eta_times_adjW = self.BDF1_eta1 * self.adjW_J_buff[idx, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:FreeDyn.nDof] = user_fcts.dLdv.T + self.BDF1_eta0_inv*(user_fcts.dLdq.T - eta_times_adjW) - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx, :]
        self.BDF_solVec_J[FreeDyn.nDof:FreeDyn.nDofConstr] = FreeDyn.MBS_Cq @ (eta_times_adjW - user_fcts.dLdq.T)
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_dense(self, FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_J_AdjSys_dense()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx
        
        # Compute adj p at time idx = 0
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_P_Sig_MU[:FreeDyn.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = FreeDyn.MBS_G_tr.T @ vec_P_Sig_MU[:FreeDyn.nDof]
        self.adjW_J_buff[self.BDF_idx_buff, :] += FreeDyn.MBS_CqvDq.T @ vec_P_Sig_MU[FreeDyn.nDof:FreeDyn.nDofConstr] 
        self.adjW_J_buff[self.BDF_idx_buff, :] += FreeDyn.MBS_Cq.T @ vec_P_Sig_MU[FreeDyn.nDofConstr:] 
        self.adjW_J_buff[self.BDF_idx_buff, :] -= eta_times_adjW
        self.adjW_J_buff[self.BDF_idx_buff, :] += user_fcts.dLdq.T
        self.adjW_J_buff[self.BDF_idx_buff, :] *= self.BDF1_eta0_inv
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_J_sparse(self, user_fcts, FreeDyn, z, deltaT):
        
        idx = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx, :] = FreeDyn.MBS_M @ self.adjP_J_buff[idx, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat(FreeDyn)
        user_fcts.get_Lagrangian_dq(FreeDyn, z)
        user_fcts.get_Lagrangian_dv(FreeDyn, z)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx] = deltaT
        self.get_BDForder1_coeffs_eta(idx) 
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:FreeDyn.nDof] = user_fcts.dLdq.T - self.BDF1_eta1 * self.adjW_J_buff[idx, :]
        self.BDF_solVec_J[FreeDyn.nDof:2*FreeDyn.nDof] = user_fcts.dLdv.T - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx, :]
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_sparse(self, FreeDyn, self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_J_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx
        
        # Compute adj p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:FreeDyn.nDof]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[FreeDyn.nDof:2*FreeDyn.nDof]
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi_dense(self, FreeDyn, deltaT):
        
        idx = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx, :, :] = FreeDyn.MBS_M @ self.adjP_Phi_buff[idx, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat(FreeDyn)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx] = deltaT
        self.get_BDForder1_coeffs_eta(idx) 
        
        # pre-compute sums of matrices
        eta_times_adjW = self.BDF1_eta1 * self.adjW_Phi_buff[idx, :, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_Phi[:FreeDyn.nDof,:] =  - self.BDF1_eta0_inv * eta_times_adjW  - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx, :, :]
        self.BDF_solVec_Phi[FreeDyn.nDof:FreeDyn.nDofConstr,:] = FreeDyn.MBS_Cq @ eta_times_adjW
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_dense(self, FreeDyn, self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_Phi_AdjSys_dense()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx
        
        # Compute adj p at time idx = 0
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_P_Sig_MU[:FreeDyn.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = FreeDyn.MBS_G_tr.T @ vec_P_Sig_MU[:FreeDyn.nDof]
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += FreeDyn.MBS_CqvDq.T @ vec_P_Sig_MU[FreeDyn.nDof:FreeDyn.nDofConstr] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += FreeDyn.MBS_Cq.T @ vec_P_Sig_MU[FreeDyn.nDofConstr:] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= eta_times_adjW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= self.BDF1_eta0_inv
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi_sparse(self, FreeDyn, deltaT):
        
        idx = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx, :, :] = FreeDyn.MBS_M @ self.adjP_Phi_buff[idx, :] 
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat(FreeDyn)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx] = deltaT
        self.get_BDForder1_coeffs_eta(idx) 
        
        # Bulid the solution vector of the Adjoint Sys        
        self.BDF_solVec_Phi[:FreeDyn.nDof,:] =  - self.BDF1_eta1 * self.adjW_Phi_buff[idx, :, :] 
        self.BDF_solVec_Phi[FreeDyn.nDof:2*FreeDyn.nDof,:] = - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx, :, :]
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_sparse(self, FreeDyn, self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_Phi_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx
        
        # Compute adj w and p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:FreeDyn.nDof]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[FreeDyn.nDof:2*FreeDyn.nDof]
# -----------------------------------------------------------------------------