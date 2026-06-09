import numpy as np

class adjGrads:
    
    def __init__(self):

        self.adjGrad_J_buff = np.zeros((2, self.num_ctrls, self.num_ctrl_gridNodes))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0].reshape(-1)
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1].reshape(-1)
        
        if self.num_xF > 0:
            ctrl_all = self.num_ctrls * self.num_ctrl_gridNodes
            self.adjGrad_Phi_buff = np.zeros((2, self.num_xF, self.num_ctrls, self.num_ctrl_gridNodes))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0].reshape(self.num_xF, ctrl_all)
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1].reshape(self.num_xF, ctrl_all)
# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, idx):
        
        self.fd_model.fetch_states_at_index(idx)
        self.fd_model.update_state_at_index(idx)
        self.fd_model.update_jacobian()
        self.buffer_MBS_fDu.update_from_dll()
        
        return self.fd_model.t
# -----------------------------------------------------------------------------        

    def adjGrad_J(self, z):
        
        """ Gradient of the cost functioncal J w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(self.num_time_steps-1)
        self.get_consistent_BC_J() 
        self.get_Lagrangian_du(z)
        dLdu_adjP_fdu = self.dLdu + self.get_adjVar_p_J().T @ self.fDu
        vec_C = self.get_vec_c(tRight/self.tF)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_J(z, deltaT)        
        self.get_Lagrangian_du(z)
        dLdu_adjP_fdu = self.dLdu + self.get_adjVar_p_J().T @ self.fDu
        vec_C = self.get_vec_c(tLeft/self.tF)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        dJdu = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
                
        """ BDF order 2 """        
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_J(z, deltaT)
            self.get_Lagrangian_du(z)
            dLdu_adjP_fdu = self.dLdu + self.get_adjVar_p_J().T @ self.fDu
            vec_C = self.get_vec_c(tLeft/self.tF)
            np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
            dJdu += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
        dJdu *= 0.5 
                
        return dJdu
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, z):
        
        """ Gradient of the final constraints Phi w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(self.num_time_steps-1)
        self.get_consistent_BC_Phi() 
        vec_C = self.get_vec_c(tRight/self.tF)
        adjP_fdu = self.get_adjVar_P_Phi().T @ self.fDu
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_Phi(deltaT)
        vec_C = self.get_vec_c(tLeft/self.tF)
        adjP_fdu = self.get_adjVar_P_Phi().T @ self.fDu
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_Phi(deltaT)
            vec_C = self.get_vec_c(tLeft/self.tF)
            adjP_fdu = self.get_adjVar_P_Phi().T @ self.fDu
            np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidu *= 0.5
                
        return dPhidu
# ----------------------------------------------------------------------------- 