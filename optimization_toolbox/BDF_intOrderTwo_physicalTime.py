class BDF_intOrderTwo:
    
    def __init__(self):
        
        self.BDF2_eta0_inv = 0.0
        self.BDF2_eta0 = 0.0
        self.BDF2_eta1 = 0.0
        self.BDF2_eta2 = 0.0
        
        if self.MBS_modeMAT_sparse:  
            self.BDForder2_singleStep_J = self.BDForder2_singleStep_J_sparse
            self.BDForder2_singleStep_Phi = self.BDForder2_singleStep_Phi_sparse
        else:
            self.BDForder2_singleStep_J = self.BDForder2_singleStep_J_dense
            self.BDForder2_singleStep_Phi = self.BDForder2_singleStep_Phi_dense

        print('BDF integration order 2 initialized')    
# -----------------------------------------------------------------------------

    def getEta_BDF2(self, idx1, idx2):
        
        diff_01 = self.BDF_diff_tau[idx1]                                       # s_{n} - s_{n-1}
        diff_12 = self.BDF_diff_tau[idx2]                                       # s_{n-1} - s_{n-2}
        diff_02 = diff_01 + diff_12                                             # s_{n} - s_{n-2}

        self.BDF2_eta0_inv = (diff_01 * diff_02) / (diff_01 + diff_02)          # inv(eta0)
        
        self.BDF2_eta0 = 1 / self.BDF2_eta0_inv                                 # eta_0
        self.BDF2_eta1 = -diff_02 / (diff_01 * diff_12)                         # eta_1
        self.BDF2_eta2 = diff_01 / (diff_02 * diff_12)                          # eta_2
# -----------------------------------------------------------------------------

    def BDForder2_singleStep_J_dense(self, z):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1

        self.BDF_J_Mp_buff[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]


        self.update_MBS_SysMat()
        self.update_userFcts_BDF(z)
        
        
        # Compute BDF coefficients
        self.getEta_BDF2(idx1, idx2) 
        
        etaTimesW = self.BDF2_eta1 * self.adjW_J_buff[idx1, :] + self.BDF2_eta2 * self.adjW_J_buff[idx2, :]
        
        
        # Bulid the solution vector of the Adjoint Sys
        
        self.BDF_solVec_J[:self.nDof] = self.dLdv.T + self.BDF2_eta0_inv*(self.dLdq.T - etaTimesW) - (self.BDF2_eta1 * self.BDF_J_Mp_buff[idx1, :] + self.BDF2_eta2 * self.BDF_J_Mp_buff[idx2, :])
        self.BDF_solVec_J[self.nDof:self.nDofConstr] = self.MBS_Cq @ (etaTimesW - self.dLdq.T)
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_dense(self.BDF2_eta0, self.BDF2_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_J_AdjSys_dense()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj p at time idx = 0
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_P_Sig_MU[:self.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_J_buff[self.BDF_idx_buff, :] -= etaTimesW
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.dLdq.T
        self.adjW_J_buff[self.BDF_idx_buff, :] *= self.BDF2_eta0_inv
# -----------------------------------------------------------------------------  

    def BDForder2_singleStep_J_sparse(self, z):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1

        self.BDF_J_Mp_buff[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]
        
        self.update_MBS_SysMat()
        self.update_userFcts_BDF(z)
        
        
        # Compute BDF coefficients
        self.getEta_BDF2(idx1, idx2)
        
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdq.T - self.BDF2_eta1 * self.adjW_J_buff[idx1, :] - self.BDF2_eta2 * self.adjW_J_buff[idx2, :] 
        self.BDF_solVec_J[self.nDof:2*self.nDof] = self.dLdv.T - self.BDF2_eta1 * self.BDF_J_Mp_buff[idx1, :] - self.BDF2_eta2 * self.BDF_J_Mp_buff[idx2, :]
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_sparse(self.BDF2_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_J_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        # Compute adj p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]
# ----------------------------------------------------------------------------- 

    def BDForder2_singleStep_Phi_dense(self):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        self.BDF_Phi_Mp_buff[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :, :]

        self.update_MBS_SysMat()
        
        
        # Compute BDF coefficients
        self.getEta_BDF2(idx1, idx2) 
        
        etaTimesW = self.BDF2_eta1 * self.adjW_Phi_buff[idx1, :, :] + self.BDF2_eta2 * self.adjW_Phi_buff[idx2, :, :]
        
        # Bulid the solution vector of the Adjoint Sys      
        self.BDF_solVec_Phi[:self.nDof, :] = -self.BDF2_eta0_inv * etaTimesW - (self.BDF2_eta1 * self.BDF_Phi_Mp_buff[idx1, :, :] + self.BDF2_eta2 * self.BDF_Phi_Mp_buff[idx2, :, :])
        self.BDF_solVec_Phi[self.nDof:self.nDofConstr,:] = self.MBS_Cq @ etaTimesW
 
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_dense(self.BDF2_eta0, self.BDF2_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_Phi_AdjSys_dense()


        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2

        
        # Compute adj p at time idx = 0
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_P_Sig_MU[:self.nDof]
        
        # Compute adj w at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= etaTimesW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= self.BDF2_eta0_inv       
# ----------------------------------------------------------------------------- 

    def BDForder2_singleStep_Phi_sparse(self):
        
        idx1 = self.BDF_idx_buff
        idx2 = 1 - idx1
        
        self.BDF_Phi_Mp_buff[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :, :]
        
        self.update_MBS_SysMat()
        
        # Compute BDF coefficients
        self.getEta_BDF2(idx1, idx2) 
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_Phi[:self.nDof,:] = - self.BDF2_eta1 * self.adjW_Phi_buff[idx1, :, :] - self.BDF2_eta2 * self.adjW_Phi_buff[idx2, :, :]
        self.BDF_solVec_Phi[self.nDof:2*self.nDof,:] = - self.BDF2_eta1 * self.BDF_Phi_Mp_buff[idx1, :, :] - self.BDF2_eta2 * self.BDF_Phi_Mp_buff[idx2, :, :]
 
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_sparse(self.BDF2_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_Phi_AdjSys_sparse()
        
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = idx2
        
        
        # Compute adj p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]
# -----------------------------------------------------------------------------