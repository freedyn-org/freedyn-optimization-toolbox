import numpy as np

class adjGrads:
    
    def __init__(self):
        
        self.adjGrad_J_uDach_buff = np.zeros((2, self.num_ctrls, self.num_ctrl_gridNodes))
        self.adjGrad_J_uDach_buff_view0 = self.adjGrad_J_uDach_buff[0].reshape(-1)
        self.adjGrad_J_uDach_buff_view1 = self.adjGrad_J_uDach_buff[1].reshape(-1)
        
        self.adjGrad_Phi_uDach_buff = np.zeros((2, self.num_xF, self.num_ctrls, self.num_ctrl_gridNodes))
        self.adjGrad_Phi_uDach_buff_view0 = self.adjGrad_Phi_uDach_buff[0].reshape(self.num_xF, self.num_ctrls*self.num_ctrl_gridNodes)
        self.adjGrad_Phi_uDach_buff_view1 = self.adjGrad_Phi_uDach_buff[1].reshape(self.num_xF, self.num_ctrls*self.num_ctrl_gridNodes)
 
# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, idx):
        
        self.fd_model.fetch_states_at_index(idx)
        self.fd_model.update_state_at_index(idx)
        self.fd_model.update_jacobian()
        self.buffer_MBS_fDu.update_from_dll()
        
        return self.fd_model.t

# -----------------------------------------------------------------------------
    
    def adjGrad_J(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradJ = np.zeros(self.num_optVars)
        
        idx_buff = 0
        
        tRight = self.adjGrad_updates(self.num_time_steps-1)
        self.get_consistent_BC_J() 
        self.get_LagrangianOCP_du(z)
        
        
        dLdu_adjP_fdu = self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Right = tRight * (dLdu_adjP_fdu @ dCdtau_uDach)
        LagrangianOCP_tF = self.get_LagrangianOCP(z) # is added at the end, otherwise multiplied by 0.5
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.BDForder1_singleStep_J(z)        
        self.get_LagrangianOCP_du(z)
        
        dLdu_adjP_fdu = self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
        
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_uDach)

        gradJ[0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradJ[1:] += deltaT * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
                
        """ BDF order 2 """        
        
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
  
            tLeft = self.adjGrad_updates(i)
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.BDForder2_singleStep_J(z)
            
            
            self.get_LagrangianOCP_du(z)
            dLdu_adjP_fdu = self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu
            vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
            
            np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_uDach)
            
            gradJ[0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
            gradJ[1:] += deltaT * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
        
        
        gradJ *= 0.5
        gradJ[0] += LagrangianOCP_tF  # is added here, otherwise multiplied by 0.5

                
        return gradJ
    
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradPhi = np.zeros([self.num_xF, self.num_optVars])
        
        idx_buff = 0

        tRight = self.adjGrad_updates(self.num_time_steps-1)
        
        self.get_consistent_BC_Phi() # this updates self.dPhidq and self.dPhidv
        
        dPhidt_tF = self.dPhidq @ self.fd_model.Qd[:, 0] + self.dPhidv @ self.fd_model.Qdd[:, 0]  # is added at the end, otherwise multiplied by 0.5y()
        
        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])        
        integrand_tF_Right = tRight * (adjP_fdu @ dCdtau_uDach)
        
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.BDForder1_singleStep_Phi()

        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
        
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_uDach)
        
        gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)   
        
        """ BDF order 2 """      
        
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
            
            tLeft = self.adjGrad_updates(i)
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.BDForder2_singleStep_Phi()
            
            adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
            vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
            
            np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_uDach)
            
            gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
            gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)
        
        gradPhi *= 0.5
        
        gradPhi[:,0] += dPhidt_tF # is added here, otherwise multiplied by 0.5
                
        return gradPhi
    
# -----------------------------------------------------------------------------    