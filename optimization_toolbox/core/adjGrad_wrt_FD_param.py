import numpy as np

from core.consistent_boundary_conditions import BC_FDOP
from core.BDF_physicalTime import BDF

class adjGrads(BC_FDOP, BDF):
    
    def __init__(self, dataOpt, FreeDyn, nPars):
        
        BC_FDOP.__init__(self, dataOpt, FreeDyn)
        BDF.__init__(self, dataOpt, FreeDyn)

        self.adjGrad_J_buff = np.zeros((2, nPars))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0]
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1]
        
        if dataOpt.num_xF > 0:
            self.adjGrad_Phi_buff = np.zeros((2, dataOpt.num_xF, nPars))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0]
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1]
# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, FreeDyn, idx):
        
        FreeDyn.API.fetch_states_at_index(idx)
        FreeDyn.API.update_state_at_index(idx)
        FreeDyn.API.update_jacobian()
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        
        return FreeDyn.API.t
# -----------------------------------------------------------------------------        

    def adjGrad_J(self, dataOpt, user_Fcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the cost functioncal J w.r.t. FreeDyn parameters 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-1)
        self.get_consistent_BC_J(FreeDyn) 
        user_Fcts.get_Lagrangian_dpars(FreeDyn, z)
        self.adjGrad_J_buff[idx_buff] = user_Fcts.dLdpars + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_J(user_Fcts, FreeDyn, z, deltaT)        
        user_Fcts.get_Lagrangian_dpars(FreeDyn, z)
        self.adjGrad_J_buff[idx_buff] = user_Fcts.dLdpars + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

        dJdpars = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
                
        """ BDF order 2 """        
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(FreeDyn, i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_J(user_Fcts, FreeDyn, z, deltaT)
            user_Fcts.get_Lagrangian_dpars(FreeDyn, z)
            self.adjGrad_J_buff[idx_buff] = user_Fcts.dLdpars + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

            dJdpars += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
        dJdpars *= 0.5 
                
        return dJdpars
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, dataOpt, user_Fcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the final constraints Phi w.r.t. FreeDyn parameters  
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-1)
        self.get_consistent_BC_Phi(user_Fcts, FreeDyn) 
        vec_C = self.get_vec_c(tRight/self.tF)
        self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        np.multiply(self.adjGrad_Phi_buff[idx_buff][:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_Phi(FreeDyn, deltaT)
        self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam

        dPhidpars = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            tLeft = self.adjGrad_updates(FreeDyn, i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_Phi(FreeDyn, deltaT)
            self.adjGrad_Phi_buff[idx_buff] = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam

            dPhidpars += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidpars *= 0.5
                
        return dPhidpars
# ----------------------------------------------------------------------------- 