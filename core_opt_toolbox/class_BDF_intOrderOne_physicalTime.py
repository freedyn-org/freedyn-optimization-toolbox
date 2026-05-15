import numpy as np
import freedyn as fd


class BDF_intOrderOne:
    
    def __init__(self):
        
        self.BDF1_eta0_inv = 0.0
        self.BDF1_eta0 = 0.0
        self.BDF1_eta1 = 0.0
        
        if self.BDF_modeMAT_sparse:
            if self.spAdjSys == 0:
                self.integrate_J_BDF1 = self.integrate_J_BDF1_3x3_sparse
                self.integrate_Phi_BDF1 = self.integrate_Phi_BDF1_3x3_sparse
            if self.spAdjSys == 1:    
                self.integrate_J_BDF1 = self.integrate_J_BDF1_4x4_sparse
                self.integrate_Phi_BDF1 = self.integrate_Phi_BDF1_4x4_sparse
        else:
            self.integrate_J_BDF1 = self.integrate_J_BDF1_dense
            self.integrate_Phi_BDF1 = self.integrate_Phi_BDF1_dense
            
        
        print('BDF integration order 1 initialized')   
        
# -----------------------------------------------------------------------------

    def getEta_BDF1(self, idx1):

        self.BDF1_eta0_inv = self.BDF_diff_tau[idx1]
        self.BDF1_eta0 = 1 / self.BDF1_eta0_inv
        self.BDF1_eta1 = -self.BDF1_eta0
            
        return None
  
# -----------------------------------------------------------------------------

    def integrate_J_BDF1_dense(self, z, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_J_Mp_buff[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]
        
        self.update_MBS_SysMat()
        self.update_userFcts_BDF(z, time)
        
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        etaTimesW = self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdv.T + self.BDF1_eta0_inv*(self.dLdq.T - etaTimesW) - self.BDF1_eta1 * self.BDF_J_Mp_buff[idx1, :]
        self.BDF_solVec_J[self.nDof:self.nDofConstr] = self.MBS_Cq @ (etaTimesW - self.dLdq.T)
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_dense(self.BDF1_eta0, self.BDF1_eta0_inv)
        
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
        self.adjW_J_buff[self.BDF_idx_buff, :] -= etaTimesW
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.dLdq.T
        self.adjW_J_buff[self.BDF_idx_buff, :] *= self.BDF1_eta0_inv

        return None           
 
# -----------------------------------------------------------------------------

    def integrate_J_BDF1_3x3_sparse(self, z, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_J_Mp_buff[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]

        self.update_MBS_SysMat()
        self.update_userFcts_BDF(z, time)
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        
        etaTimesW = self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdv.T + self.BDF1_eta0_inv*(self.dLdq.T - etaTimesW) - self.BDF1_eta1 * self.BDF_J_Mp_buff[idx1, :]
        self.BDF_solVec_J[self.nDof:self.nDofConstr] = self.MBS_Cq @ (etaTimesW - self.dLdq.T)
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_3x3_sparse(self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_J_AdjSys_sparse()
        
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        
        # Compute adj p at time idx = 0
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_P_Sig_MU[:self.nDof]
        
        
        # Compute adj w at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_J_buff[self.BDF_idx_buff, :] -= etaTimesW
        self.adjW_J_buff[self.BDF_idx_buff, :] += self.dLdq.T
        self.adjW_J_buff[self.BDF_idx_buff, :] *= self.BDF1_eta0_inv

        return None           
 
# -----------------------------------------------------------------------------

    def integrate_J_BDF1_4x4_sparse(self, z, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_J_Mp_buff[idx1, :] = self.MBS_M @ self.adjP_J_buff[idx1, :]

        self.update_MBS_SysMat()
        self.update_userFcts_BDF(z, time)
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_J[:self.nDof] = self.dLdq.T - self.BDF1_eta1 * self.adjW_J_buff[idx1, :]
        self.BDF_solVec_J[self.nDof:2*self.nDof] = self.dLdv.T - self.BDF1_eta1 * self.BDF_J_Mp_buff[idx1, :]
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_4x4_sparse(self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_J_AdjSys_sparse()
        
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        
        # Compute adj p at time idx = 0
        self.adjW_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_J_buff[self.BDF_idx_buff, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]
        

        return None           
 
# ----------------------------------------------------------------------------- 

    def integrate_Phi_BDF1_dense(self, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_Phi_Mp_buff[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :]
        
        self.update_MBS_SysMat()
        
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        etaTimesW = self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        
        # Bulid the solution vector of the Adjoint Sys
        self.BDF_solVec_Phi[:self.nDof,:] =  - self.BDF1_eta0_inv * etaTimesW  - self.BDF1_eta1 * self.BDF_Phi_Mp_buff[idx1, :, :]
        self.BDF_solVec_Phi[self.nDof:self.nDofConstr,:] = self.MBS_Cq @ etaTimesW
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_dense(self.BDF1_eta0, self.BDF1_eta0_inv)
        
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
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= etaTimesW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= self.BDF1_eta0_inv

        return None
           
# ----------------------------------------------------------------------------- 

    def integrate_Phi_BDF1_3x3_sparse(self, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_Phi_Mp_buff[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :]
        
        self.update_MBS_SysMat()
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        etaTimesW = self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :]
        
        # Bulid the solution vector of the Adjoint Sys        
        self.BDF_solVec_Phi[:self.nDof,:] =  - self.BDF1_eta0_inv * etaTimesW  - self.BDF1_eta1 * self.BDF_Phi_Mp_buff[idx1, :, :]
        self.BDF_solVec_Phi[self.nDof:self.nDofConstr,:] = self.MBS_Cq @ etaTimesW
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_3x3_sparse(self.BDF1_eta0, self.BDF1_eta0_inv)
        
        # solve Matrix-Vektor Equation
        vec_P_Sig_MU = self.solve_Phi_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        
        # Compute adj p at time idx = 0
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_P_Sig_MU[:self.nDof]


        # Compute adj w at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = self.MBS_G_tr.T @ vec_P_Sig_MU[:self.nDof]
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_CqvDq.T @ vec_P_Sig_MU[self.nDof:self.nDofConstr] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] += self.MBS_Cq.T @ vec_P_Sig_MU[self.nDofConstr:] 
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] -= etaTimesW
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] *= self.BDF1_eta0_inv


        return None           
 
# ----------------------------------------------------------------------------- 

    def integrate_Phi_BDF1_4x4_sparse(self, time):
        
        idx1 = self.BDF_idx_buff
        
        self.BDF_Phi_Mp_buff[idx1, :, :] = self.MBS_M @ self.adjP_Phi_buff[idx1, :]
        
        self.update_MBS_SysMat()
        
        # Compute BDF coefficients
        self.getEta_BDF1(idx1) 
        
        
        # Bulid the solution vector of the Adjoint Sys        
        self.BDF_solVec_Phi[:self.nDof,:] =  - self.BDF1_eta1 * self.adjW_Phi_buff[idx1, :, :] 
        self.BDF_solVec_Phi[self.nDof:2*self.nDof,:] = - self.BDF1_eta1 * self.BDF_Phi_Mp_buff[idx1, :, :]
        
        
        # get coeff. Matrix of adjoint system
        self.get_coeffMat_AdjSys_4x4_sparse(self.BDF1_eta0)
        
        # solve Matrix-Vektor Equation
        vec_W_P_Sig_MU = self.solve_Phi_AdjSys_sparse()
        
        # idx shift - use memory of idx = 2 for idx = 0
        self.BDF_idx_buff = 1 - idx1
        
        
        # Compute adj p at time idx = 0
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[:self.nDof]
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = vec_W_P_Sig_MU[self.nDof:2*self.nDof]


        return None           
 
# -----------------------------------------------------------------------------