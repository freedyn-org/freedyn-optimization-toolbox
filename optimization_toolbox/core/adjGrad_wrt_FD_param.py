import numpy as np

class adjGrads:
    
    def __init__(self):

        self.adjGrad_J_buff = np.zeros((2, self.num_pars))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0]
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1]
        
        if self.num_xF > 0:
            self.adjGrad_Phi_buff = np.zeros((2, self.num_xF, self.num_pars))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0]
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1]
# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, idx):
        
        self.fd_model.fetch_states_at_index(idx)
        self.fd_model.update_state_at_index(idx)
        self.fd_model.update_jacobian()
        self.buffer_MBS_dVecForce_dFDparam.update_from_dll()
        
        return self.fd_model.t
# -----------------------------------------------------------------------------        

    def adjGrad_J(self, z):
        
        """ Gradient of the cost functioncal J w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(self.num_time_steps-1)
        self.get_consistent_BC_J() 
        self.get_Lagrangian_dpars(z)
        self.adjGrad_J_buff[idx_buff] = self.dLdpars + self.get_adjVar_p_J().T @ self.dVecForce_dFDparam
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_J(z, deltaT)        
        self.get_Lagrangian_dpars(z)
        self.adjGrad_J_buff[idx_buff] = self.dLdpars + self.get_adjVar_p_J().T @ self.dVecForce_dFDparam
        
        dJdu = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
                
        """ BDF order 2 """        
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_J(z, deltaT)
            self.get_Lagrangian_dpars(z)
            self.adjGrad_J_buff[idx_buff] = self.dLdpars + self.get_adjVar_p_J().T @ self.dVecForce_dFDparam
            
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
        self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ self.dVecForce_dFDparam

        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(self.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_Phi(deltaT)
        self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ self.dVecForce_dFDparam

        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(self.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_Phi(deltaT)
            self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ self.dVecForce_dFDparam
            
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidu *= 0.5
                
        return dPhidu
# ----------------------------------------------------------------------------- 