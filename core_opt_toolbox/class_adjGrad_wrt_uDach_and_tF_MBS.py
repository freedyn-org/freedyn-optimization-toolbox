import numpy as np

class adjGrads:
    
    def __init__(self):
        
        self.adjGrad_J_uDach_buff = np.zeros((2, self.numCtrls, self.numNodes))
        self.adjGrad_J_uDach_buff_view0 = self.adjGrad_J_uDach_buff[0].reshape(-1)
        self.adjGrad_J_uDach_buff_view1 = self.adjGrad_J_uDach_buff[1].reshape(-1)
        
        self.adjGrad_Phi_uDach_buff = np.zeros((2, self.num_xF, self.numCtrls, self.numNodes))
        self.adjGrad_Phi_uDach_buff_view0 = self.adjGrad_Phi_uDach_buff[0].reshape(self.num_xF, self.numCtrls*self.numNodes)
        self.adjGrad_Phi_uDach_buff_view1 = self.adjGrad_Phi_uDach_buff[1].reshape(self.num_xF, self.numCtrls*self.numNodes)
 
# -----------------------------------------------------------------------------
    
    def adjGrad_J_singleBDFstep(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradJ = np.zeros(self.numOptVar)
        
        idx_buff = 0
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        self.get_consistent_BC_J() 
        self.get_LagrangianOCP_du(z, tRight)
        self.buffer_MBS_fDu.update_from_dll()
        
        dLdu_adjP_fdu = self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Right = tRight * (dLdu_adjP_fdu @ dCdtau_uDach)
        LagrangianOCP_tF = self.get_LagrangianOCP(z) # is added at the end, otherwise multiplied by 0.5
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-2)
        tLeft = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-2)
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-2)
        self.fd_model.update_jacobian()
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.singleStep_BDForder_one_J(z, tLeft)        
        self.get_LagrangianOCP_du(z, tLeft)
        self.buffer_MBS_fDu.update_from_dll()
        
        dLdu_adjP_fdu = self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
        
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_uDach)

        gradJ[0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradJ[1:] += deltaT * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
                
        """ BDF order 2 """        
        
        for i in range(self.dyn_numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
            
  
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.singleStep_BDForder_two_J(z, tLeft)
            
            
            self.get_LagrangianOCP_du(z,tLeft)
            self.buffer_MBS_fDu.update_from_dll()
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
    
    def adjGrad_Phi_singleBDFstep(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradPhi = np.zeros([self.num_xF, self.numOptVar])
        
        idx_buff = 0

        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        
        self.get_consistent_BC_Phi() # this updates self.dPhidq and self.dPhidv
        
        qD_tF = self.fd_model.Qd[:, 0].copy()
        qDD_tF = self.fd_model.Qdd[:, 0].copy()
        dPhidt_tF = self.dPhidq @ qD_tF + self.dPhidv @ qDD_tF  # is added at the end, otherwise multiplied by 0.5y()
        
        
        self.buffer_MBS_fDu.update_from_dll()
        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.einsum('ij, l->ijl', adjP_fdu, vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])        
        integrand_tF_Right = tRight * (adjP_fdu @ dCdtau_uDach)
        
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-2)
        tLeft = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-2)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-2)
        self.fd_model.update_jacobian()
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.singleStep_BDForder_one_Phi(tLeft)
        self.buffer_MBS_fDu.update_from_dll()
        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
        
        np.einsum('ij, l->ijl', adjP_fdu, vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_uDach)
        
        gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)   
        
        """ BDF order 2 """      
        
        for i in range(self.dyn_numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
            
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.singleStep_BDForder_two_Phi(tLeft)
            
            self.buffer_MBS_fDu.update_from_dll()
            adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
            vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
            
            np.einsum('ij, l->ijl', adjP_fdu, vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_uDach)
            
            gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
            gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)
        
        gradPhi *= 0.5
        
        gradPhi[:,0] += dPhidt_tF # is added here, otherwise multiplied by 0.5
                
        return gradPhi
    
# ----------------------------------------------------------------------------- 

    def adjGrad_J_BDFthenGrad(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradJ = np.zeros(self.numOptVar)

        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        
        adj_p = self.apply_BDF_J(self.dyn_numTimeSteps, tRight, z)
        
        """  Trapez rule  """
        idx_buff = 0
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        
        self.get_LagrangianOCP_du(z, tRight)
        self.buffer_MBS_fDu.update_from_dll()
        dLdu_adjP_fdu = self.dLdu + adj_p[:,-1].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Right = tRight * (dLdu_adjP_fdu @ dCdtau_uDach)
        
        LagrangianOCP_tF = self.get_LagrangianOCP(z) # is added at the end, otherwise multiplied by 0.5
        
        for i in range(self.dyn_numTimeSteps-2, -1, -1):
            
            idx_buff = 1 - idx_buff
            
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            self.get_LagrangianOCP_du(z,tLeft)
            self.buffer_MBS_fDu.update_from_dll()       
            dLdu_adjP_fdu = self.dLdu + adj_p[:,i].T @ self.fDu
            vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
            
            np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_uDach)


            gradJ[0] -= (tRight - tLeft) * (integrand_tF_Left + integrand_tF_Right)
            gradJ[1:] += (tRight - tLeft) * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
            
            tRight = tLeft
            integrand_tF_Right = integrand_tF_Left
            
        gradJ *= 0.5
        gradJ[0] += LagrangianOCP_tF  # is added here, otherwise multiplied by 0.5
                
        return gradJ
    
# ----------------------------------------------------------------------------- 
    
    def adjGrad_Phi_BDFthenGrad(self, z):
        
        vecC_dtF_invariant = self.init_invariant_vec_c_dtF()
        
        gradPhi = np.zeros([self.num_xF, self.numOptVar])   

        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1) 
        self.fd_model.update_jacobian()
        
        qD_tF = self.fd_model.Qd[:, 0].copy()
        qDD_tF = self.fd_model.Qdd[:, 0].copy()
        self.get_Phi_dq()
        self.get_Phi_dv()
        dPhidt_tF = self.dPhidq @ qD_tF + self.dPhidv @ qDD_tF  # is added at the end, otherwise multiplied by 0.5
        
        adj_P = self.apply_BDF_Phi(self.num_xF, self.dyn_numTimeSteps, tRight)
                
        
        """  Trapez rule  """  
        idx_buff = 0
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        self.buffer_MBS_fDu.update_from_dll()
        adjP_fdu = adj_P[:,:,-1].T @ self.fDu
        vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tRight/self.tF, vecC_dtF_invariant)
        
        np.einsum('ij, l->ijl', adjP_fdu, vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])   
        integrand_tF_Right = tRight * (adjP_fdu @ dCdtau_uDach)

        
        for i in range(self.dyn_numTimeSteps-2, -1, -1):
            idx_buff = 1 - idx_buff
            self.fd_model.fetch_states_at_index(i) 
            tLeft = self.fd_model.get_time_at_index(i)
            
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            self.buffer_MBS_fDu.update_from_dll()
            adjP_fdu = adj_P[:,:,i].T @ self.fDu
            vec_C, dCdtau_uDach = self.get_vec_c_AND_dCdtau_uDach(tLeft/self.tF, vecC_dtF_invariant)
            
            np.einsum('ij, l->ijl', adjP_fdu, vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_uDach)
            
            gradPhi[:,0] -= (tRight - tLeft) * (integrand_tF_Left + integrand_tF_Right)
            gradPhi[:,1:] += (tRight - tLeft) * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)
            
            
            tRight = tLeft
            integrand_tF_Right = integrand_tF_Left
            
        gradPhi *= 0.5
        
        gradPhi[:,0] += dPhidt_tF # is added here, otherwise multiplied by 0.5
    
        
        return gradPhi