import numpy as np

class adjGrads:
    
    def __init__(self):
        
        self.adjGrad_J_buff = np.zeros((2, self.numCtrls, self.numNodes))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0].reshape(-1)
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1].reshape(-1)
        
        
        self.adjGrad_Phi_buff = np.zeros((2, self.num_xF, self.numCtrls, self.numNodes))
        self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0].reshape(self.num_xF, self.numCtrls*self.numNodes)
        self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1].reshape(self.num_xF, self.numCtrls*self.numNodes)

# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, idx):
        
        self.fd_model.fetch_states_at_index(idx)
        self.fd_model.update_state_at_index(idx)
        self.fd_model.update_jacobian()
        self.buffer_MBS_fDu.update_from_dll()
        
        return self.fd_model.t

# -----------------------------------------------------------------------------        

    def adjGrad_J(self, z):
        
        idx_buff = 0
        
        tRight = self.adjGrad_updates(self.numTimeSteps-1)
        
        self.get_consistent_BC_J() 
        self.get_LagrangianOCP_du(z)

        vec_C = self.get_vec_c(tRight/self.tF)
        np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        
        tLeft = self.adjGrad_updates(self.numTimeSteps-2)
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.BDForder1_singleStep_J(z)        
        self.get_LagrangianOCP_du(z)

        vec_C = self.get_vec_c(tLeft/self.tF)
        
        np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
    

        dJdu = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
        
                
        """ BDF order 2 """        
        
        for i in range(self.numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            
            tRight = tLeft
            
            tLeft = self.adjGrad_updates(i)
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.BDForder2_singleStep_J(z)
                        
            self.get_LagrangianOCP_du(z)

            vec_C = self.get_vec_c(tLeft/self.tF)
            
            np.outer(self.dLdu + self.adjP_J_buff[self.BDF_idx_buff, :].T @ self.fDu, vec_C, out = self.adjGrad_J_buff[idx_buff])
    
            dJdu += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
        dJdu *= 0.5 

                
        return dJdu
    
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, z):
        
        idx_buff = 0

        tRight = self.adjGrad_updates(self.numTimeSteps-1)

        self.get_consistent_BC_Phi() 

        vec_C = self.get_vec_c(tRight/self.tF)
        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        
        tLeft = self.adjGrad_updates(self.numTimeSteps-2)
        
        deltaT = tRight - tLeft
        self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
        
        self.BDForder1_singleStep_Phi()

        vec_C = self.get_vec_c(tLeft/self.tF)
        
        adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])

        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        
        """ BDF order 2 """      
        
        for i in range(self.numTimeSteps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            
            tLeft = self.adjGrad_updates(i)
            
            deltaT = tRight - tLeft
            self.BDF_diff_tau[self.BDF_idx_buff] = deltaT
            
            self.BDForder2_singleStep_Phi()

            vec_C = self.get_vec_c(tLeft/self.tF)
            adjP_fdu = self.adjP_Phi_buff[self.BDF_idx_buff,:,:].T @ self.fDu
            np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
    
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        dPhidu *= 0.5
                
        return dPhidu
    
# ----------------------------------------------------------------------------- 