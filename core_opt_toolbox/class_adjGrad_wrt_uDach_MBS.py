import numpy as np

class adjGrads:
    
    def __init__(self):
        
        self.adjGrad_J_buff = np.zeros((2, self.numCtrls, self.numNodes))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0].reshape(-1)
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1].reshape(-1)
        
        
        self.adjGrad_Phi_buff = np.zeros((2, self.num_xF, self.numCtrls, self.numNodes))
        self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0].reshape(self.num_xF, self.numCtrls*self.numNodes)
        self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1].reshape(self.num_xF, self.numCtrls*self.numNodes)
    

    def adjGrad_J_singleBDFstep(self, z):
        
        idx_buff = 0
        
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        self.get_consistent_BC_J() 
        self.get_LagrangianOCP_du(z, tRight)
        self.buffer_MBS_fDu.update_from_dll()
        vec_C = self.get_vec_c(tRight/self.tF)
        np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
        
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
        vec_C = self.get_vec_c(tLeft/self.tF)
        
        np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
    

        dJdu = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
        
                
        """ BDF order 2 """        
        
        for i in range(self.dyn_numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            
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
            vec_C = self.get_vec_c(tLeft/self.tF)
            
            np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
    
            dJdu += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
        dJdu *= 0.5 

                
        return dJdu
    
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi_singleBDFstep(self, z):
        
        idx_buff = 0

        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1) 
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        self.get_consistent_BC_Phi() 
        self.buffer_MBS_fDu.update_from_dll()
        vec_C = self.get_vec_c(tRight/self.tF)
        np.einsum('ij, l->ijl', self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu, vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        
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
        vec_C = self.get_vec_c(tLeft/self.tF)
        np.einsum('ij, l->ijl', self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu, vec_C, out = self.adjGrad_Phi_buff[idx_buff])

        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        
        """ BDF order 2 """      
        
        for i in range(self.dyn_numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.singleStep_BDForder_two_Phi(tLeft)
            
            self.buffer_MBS_fDu.update_from_dll()
            vec_C = self.get_vec_c(tLeft/self.tF)
            
            np.einsum('ij, l->ijl', self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu, vec_C, out = self.adjGrad_Phi_buff[idx_buff])
    
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        dPhidu *= 0.5
                
        return dPhidu
    
# ----------------------------------------------------------------------------- 

    def adjGrad_J_BDFthenGrad(self, z):
        
        dJdu = np.zeros(self.numCtrls*self.numNodes)

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
        vec_C = self.get_vec_c(tRight/self.tF)
        np.outer(self.dLdu + adj_p[:,-1].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
        
        for i in range(self.dyn_numTimeSteps-2, -1, -1):
            
            idx_buff = 1 - idx_buff
            
            self.fd_model.fetch_states_at_index(i) 
            tLeft = self.fd_model.get_time_at_index(i)
            
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            self.get_LagrangianOCP_du(z,tLeft)
            self.buffer_MBS_fDu.update_from_dll()       
            vec_C = self.get_vec_c(tLeft/self.tF)
            np.outer(self.dLdu + adj_p[:,i].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
    
            dJdu += (tRight - tLeft) * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
            tRight = tLeft
            
        dJdu *= 0.5
                
        return dJdu
    
# ----------------------------------------------------------------------------- 
    
    def adjGrad_Phi_BDFthenGrad(self, z):
        
        dPhidu = np.zeros([self.num_xF, self.numCtrls*self.numNodes])   

        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1) 
        self.fd_model.update_jacobian()
        
        adj_P = self.apply_BDF_Phi(self.num_xF, self.dyn_numTimeSteps, tRight)
                
        
        """  Trapez rule  """  
        idx_buff = 0
        self.buffer_MBS_fDu.update_from_dll()
        self.fd_model.fetch_states_at_index(self.dyn_numTimeSteps-1)
        tRight = self.fd_model.get_time_at_index(self.dyn_numTimeSteps-1)
        
        self.fd_model.update_state_at_index(self.dyn_numTimeSteps-1)
        self.fd_model.update_jacobian()
        vec_C = self.get_vec_c(tRight/self.tF)
        np.einsum('ij, l->ijl', adj_P[:,:,-1].T @ self.fDu, vec_C, out = self.adjGrad_Phi_buff[idx_buff])    
        
        for i in range(self.dyn_numTimeSteps-2, -1, -1):
            idx_buff = 1 - idx_buff
            
            self.fd_model.fetch_states_at_index(i)
            tLeft = self.fd_model.get_time_at_index(i)
            
            
            self.fd_model.update_state_at_index(i)
            self.fd_model.update_jacobian()
            
            self.buffer_MBS_fDu.update_from_dll()
            vec_C = self.get_vec_c(tLeft/self.tF)
            np.einsum('ij, l->ijl', adj_P[:,:,i].T @ self.fDu, vec_C, out = self.adjGrad_Phi_buff[idx_buff])
            dPhidu += (tRight - tLeft) * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)

            tRight = tLeft
            
        dPhidu *= 0.5
    
        
        return dPhidu