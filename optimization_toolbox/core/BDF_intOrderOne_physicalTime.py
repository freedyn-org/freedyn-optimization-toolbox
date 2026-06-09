import core.BDF_coeffMat as coeffMat

class BDF_intOrderOne:
    
    def __init__(self):
        
        self.BDF1_eta0_inv = 0.0
        self.BDF1_eta0 = 0.0
        self.BDF1_eta1 = 0.0
        
        if self.MBS_modeMAT_sparse:    
            self.BDForder1_singleStep_J = self.BDForder1_singleStep_J_sparse
            self.BDForder1_singleStep_Phi = self.BDForder1_singleStep_Phi_sparse
        else:
            self.BDForder1_singleStep_J = self.BDForder1_singleStep_J_dense
            self.BDForder1_singleStep_Phi = self.BDForder1_singleStep_Phi_dense   
        
        print('BDF integration order 1 initialized')   
# -----------------------------------------------------------------------------

    def get_BDForder1_coeffs_eta(self, idx1):

        self.BDF1_eta0_inv = self.BDF_diff_tau[idx1]
        self.BDF1_eta0 = 1 / self.BDF1_eta0_inv
        self.BDF1_eta1 = -self.BDF1_eta0 
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_J_dense(self, z, deltaT):
        
        idx1 = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat()
        self.get_Lagrangian_dq(z)
        self.get_Lagrangian_dv(z)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx1] = deltaT
        self.get_BDForder1_coeffs_eta(idx1) 
        
        # pre-compute sums of matrices
        eta_times_adjW = self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdv.T + self.BDF1_eta0_inv*(self.dLdq.T - eta_times_adjW) - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx1, :]
        self.BDF_solVec_J[self.nDof:self.nDofConstr] = self.MBS_Cq @ (eta_times_adjW - self.dLdq.T)
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_dense(self, self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_J_AdjSys_dense()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        # Compute adj p at time idx = 0
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_P_Sig_MU[:self.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_J_buff[self.BDF_idx_buff, :] -= eta_times_adjW
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.dLdq.T
        self.adjW_J_buff[self.BDF_idx_buff, :] *= self.BDF1_eta0_inv
# -----------------------------------------------------------------------------

    def BDForder1_singleStep_J_sparse(self, z, deltaT):
        
        idx1 = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_J_buff_M_times_p[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat()
        self.get_Lagrangian_dq(z)
        self.get_Lagrangian_dv(z)
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx1] = deltaT
        self.get_BDForder1_coeffs_eta(idx1) 
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdq.T - self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        self.BDF_solVec_J[self.nDof:2*self.nDof] = self.dLdv.T - self.BDF1_eta1 * self.BDF_J_buff_M_times_p[idx1, :]
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_sparse(self, self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_J_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        # Compute adj p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi_dense(self, deltaT):
        
        idx1 = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :]
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat()
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx1] = deltaT
        self.get_BDForder1_coeffs_eta(idx1) 
        
        # pre-compute sums of matrices
        eta_times_adjW = self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_Phi[:self.nDof,:] =  - self.BDF1_eta0_inv * eta_times_adjW  - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :]
        self.BDF_solVec_Phi[self.nDof:self.nDofConstr,:] = self.MBS_Cq @ eta_times_adjW
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_dense(self, self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_Phi_AdjSys_dense()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        # Compute adj p at time idx = 0
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_P_Sig_MU[:self.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= eta_times_adjW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= self.BDF1_eta0_inv
# ----------------------------------------------------------------------------- 

    def BDForder1_singleStep_Phi_sparse(self, deltaT):
        
        idx1 = self.BDF_idx_buff
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.BDF_Phi_buff_M_times_P[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :] 
        
        # Updates at BDF time step s_n
        self.update_MBS_SysMat()
        
        # Compute BDF coefficients
        self.BDF_diff_tau[idx1] = deltaT
        self.get_BDForder1_coeffs_eta(idx1) 
        
        # Bulid the solution vector of the Adjoint Sys        
        self.BDF_solVec_Phi[:self.nDof,:] =  - self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :] 
        self.BDF_solVec_Phi[self.nDof:2*self.nDof,:] = - self.BDF1_eta1 * self.BDF_Phi_buff_M_times_P[idx1, :, :]
        
        # get coeff. Matrix of adjoint system
        coeffMat.update_coeffMat_AdjSys_sparse(self, self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_Phi_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        # Compute adj w and p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]
# -----------------------------------------------------------------------------