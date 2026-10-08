import numpy as np

from core.adjoint_system_settings import set_up_consistent_BC_FDOP
from core.adjoint_system_settings import set_up_BDF_layout_sys_mat

class adjGrads_OCP():
    
    def __init__(self, dataOpt, FreeDyn, Ctrl):
        
        self.BC_FDOP = set_up_consistent_BC_FDOP(dataOpt, FreeDyn)
        self.BDF = set_up_BDF_layout_sys_mat(dataOpt, FreeDyn)

        self.adjGrad_J_buff = np.zeros((2, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0].reshape(-1)
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1].reshape(-1)
        
        if dataOpt.num_xF > 0:
            ctrl_all = Ctrl.num_ctrls * Ctrl.num_grid_nodes
            self.adjGrad_Phi_buff = np.zeros((2, dataOpt.num_xF, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0].reshape(dataOpt.num_xF, ctrl_all)
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1].reshape(dataOpt.num_xF, ctrl_all)
# -----------------------------------------------------------------------------        

    def adjGrad_J(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the cost functioncal J w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        tRight = FreeDyn.API.t
        self.BC_FDOP.get_consistent_BC_J(self.BDF, FreeDyn) 
        UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
        vec_C = Ctrl.get_vec_c(tRight/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_J(UserFcts, FreeDyn, z, deltaT)        
        UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
        vec_C = Ctrl.get_vec_c(tLeft/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
        dJdu = deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
                
        """ BDF order 2 """        
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            FreeDyn.fetch_and_update_states_at_index(i)
            tLeft = FreeDyn.API.t
            deltaT = tRight - tLeft
            self.BDF.BDForder2_singleStep_J(UserFcts, FreeDyn, z, deltaT)
            UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
            FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
            dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
            vec_C = Ctrl.get_vec_c(tLeft/dataOpt.final_time)
            np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff[idx_buff])
            dJdu += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
        dJdu *= 0.5 
                
        return dJdu
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the final constraints Phi w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        tRight = FreeDyn.API.t
        self.BC_FDOP.get_consistent_BC_Phi(self.BDF, UserFcts, FreeDyn) 
        vec_C = Ctrl.get_vec_c(tRight/dataOpt.final_time)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_Phi(FreeDyn, deltaT)
        vec_C = Ctrl.get_vec_c(tLeft/dataOpt.final_time)
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            FreeDyn.fetch_and_update_states_at_index(i)
            tLeft = FreeDyn.API.t
            deltaT = tRight - tLeft
            self.BDF.BDForder2_singleStep_Phi(FreeDyn, deltaT)
            vec_C = Ctrl.get_vec_c(tLeft/dataOpt.final_time)
            FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
            adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
            np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff[idx_buff])
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidu *= 0.5
                
        return dPhidu
# ----------------------------------------------------------------------------- 