import numpy as np

from .consistent_boundary_conditions import CoreBC
from .BDF_sparse_physical_time import SparseBDF
from .BDF_dense_physical_time import DenseBDF

class adjGrads_Param():
    
    def __init__(self, dataOpt, FreeDyn, Ctrl):
        
        self.BC_FDOP = CoreBC.create(dataOpt, FreeDyn)
        
        if FreeDyn.MBS_modeMAT_sparse:
            self.BDF = SparseBDF(dataOpt, FreeDyn)
        else:
            self.BDF =  DenseBDF(dataOpt, FreeDyn)

        self.adjGrad_J_buff = np.zeros((2, FreeDyn.num_FD_pars))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0]
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1]
        
        if dataOpt.num_xF > 0:
            self.adjGrad_Phi_buff = np.zeros((2, dataOpt.num_xF, FreeDyn.num_FD_pars))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0]
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1]
# -----------------------------------------------------------------------------        

    def adjGrad_J(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the cost functioncal J w.r.t. FreeDyn parameters 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        tRight = FreeDyn.API.t
        self.BC_FDOP.get_consistent_BC_J(self.BDF, FreeDyn) 
        UserFcts.get_lagrangian_dpars(FreeDyn, z)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        self.adjGrad_J_buff[idx_buff] = UserFcts.dLdpars + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_J(UserFcts, FreeDyn, z, deltaT)        
        UserFcts.get_lagrangian_dpars(FreeDyn, z)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        self.adjGrad_J_buff[idx_buff] = UserFcts.dLdpars + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

        dJdpars = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
                
        """ BDF order 2 """        
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            FreeDyn.fetch_and_update_states_at_index(i)
            tLeft = FreeDyn.API.t
            deltaT = tRight - tLeft
            self.BDF.BDForder2_singleStep_J(UserFcts, FreeDyn, z, deltaT)
            UserFcts.get_lagrangian_dpars(FreeDyn, z)
            FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
            self.adjGrad_J_buff[idx_buff] = UserFcts.dLdpars + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam

            dJdpars += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
        dJdpars *= 0.5 
                
        return dJdpars
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the final constraints Phi w.r.t. FreeDyn parameters  
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        tRight = FreeDyn.API.t
        self.BC_FDOP.get_consistent_BC_Phi(self.BDF, UserFcts, FreeDyn) 
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        self.adjGrad_Phi_buff[idx_buff] = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_Phi(FreeDyn, deltaT)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        self.adjGrad_Phi_buff[idx_buff] = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam

        dPhidpars = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            FreeDyn.fetch_and_update_states_at_index(i)
            tLeft = FreeDyn.API.t
            deltaT = tRight - tLeft
            self.BDF.BDForder2_singleStep_Phi(FreeDyn, deltaT)
            FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
            self.adjGrad_Phi_buff[idx_buff] = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam

            dPhidpars += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidpars *= 0.5
                
        return dPhidpars
# ----------------------------------------------------------------------------- 