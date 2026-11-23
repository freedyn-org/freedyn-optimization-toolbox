import numpy as np

from .consistent_boundary_conditions import CoreBC
from .BDF_core import BDF

class adjGrads_OCP():
    
    def __init__(self, FreeDyn, Ctrl, num_xF):
        
        self.BC_FDOP = CoreBC.create(FreeDyn, num_xF)
        self.BDF = BDF.create(FreeDyn, num_xF)
        
        self.adjGrad_J_buff = np.zeros((2, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
        self.adjGrad_J_buff_view0 = self.adjGrad_J_buff[0].reshape(-1)
        self.adjGrad_J_buff_view1 = self.adjGrad_J_buff[1].reshape(-1)

        # self.adjGrad_J_buff = np.zeros((Ctrl.num_ctrls, Ctrl.num_grid_nodes))
        # self.adjGrad_J_buff_view = self.adjGrad_J_buff.reshape(-1)
        
        if num_xF > 0:
            ctrl_all = Ctrl.num_ctrls * Ctrl.num_grid_nodes
            self.adjGrad_Phi_buff = np.zeros((2, num_xF, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
            self.adjGrad_Phi_buff_view0 = self.adjGrad_Phi_buff[0].reshape(num_xF, ctrl_all)
            self.adjGrad_Phi_buff_view1 = self.adjGrad_Phi_buff[1].reshape(num_xF, ctrl_all)
            # self.adjGrad_Phi_buff = np.zeros((dataOpt.num_xF, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
            # self.adjGrad_Phi_buff_view = self.adjGrad_Phi_buff.reshape(dataOpt.num_xF, ctrl_all)

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
        FreeDyn.update_dForce_dFDparam()
        dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.SysMat.dForce_dFDparam
        Ctrl.update_vecC(tRight/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_J_buff[idx_buff])
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_J(UserFcts, FreeDyn, z, deltaT)        
        UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
        FreeDyn.update_dForce_dFDparam()
        dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.SysMat.dForce_dFDparam
        Ctrl.update_vecC(tLeft/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_J_buff[idx_buff])
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
            FreeDyn.update_dForce_dFDparam()
            dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.SysMat.dForce_dFDparam
            Ctrl.update_vecC(tLeft/dataOpt.final_time)
            np.outer(dLdu_adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_J_buff[idx_buff])
            dJdu += deltaT * (self.adjGrad_J_buff_view0 + self.adjGrad_J_buff_view1)
            
        dJdu *= 0.5 
                
        return dJdu
# -----------------------------------------------------------------------------        

    # def adjGrad_J(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
    #     """ Gradient of the cost functioncal J w.r.t. uDach 
    #         Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
    #     """ t = t_f / init BDF routine """
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
    #     t_i = FreeDyn.API.t
    #     self.BC_FDOP.get_consistent_BC_J(self.BDF, FreeDyn) 
    #     UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #     np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff)
        
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
    #     t_i_min = FreeDyn.API.t
        
    #     dJdu = (t_i - t_i_min) * self.adjGrad_J_buff_view
        
        
    #     """ BDF order 1 """
    #     t_i_plus = t_i
    #     t_i = t_i_min
    #     BDF_delta_t = t_i_plus - t_i
    #     self.BDF.BDForder1_singleStep_J(UserFcts, FreeDyn, z, BDF_delta_t)        
    #     UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #     np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff)
        
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-3)
    #     t_i_min = FreeDyn.API.t
    #     dJdu += (t_i_plus - t_i_min) * self.adjGrad_J_buff_view
        
                
    #     """ BDF order 2 """        
    #     for i in range(FreeDyn.num_time_steps-3, 0, -1):
    #         t_i_plus = t_i
    #         t_i = t_i_min
    #         BDF_delta_t = t_i_plus - t_i
    #         self.BDF.BDForder2_singleStep_J(UserFcts, FreeDyn, z, BDF_delta_t)
    #         UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
    #         FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #         dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
    #         vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #         np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff)
            
    #         FreeDyn.fetch_and_update_states_at_index(i-1)
    #         t_i_min = FreeDyn.API.t
    #         dJdu += (t_i_plus - t_i_min) * self.adjGrad_J_buff_view
            
        
    #     BDF_delta_t = t_i - t_i_min
    #     self.BDF.BDForder2_singleStep_J(UserFcts, FreeDyn, z, BDF_delta_t)
    #     UserFcts.get_lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     dLdu_adjP_fdu = UserFcts.dLdu + self.BDF.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i_min/dataOpt.final_time)
    #     np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_buff)  
    #     dJdu += (t_i - t_i_min) * self.adjGrad_J_buff_view         
            
    #     dJdu *= 0.5 
                
    #     return dJdu
# -----------------------------------------------------------------------------
    
    def adjGrad_Phi(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the final constraints Phi w.r.t. uDach 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
        tRight = FreeDyn.API.t
        self.BC_FDOP.get_consistent_BC_Phi(self.BDF, UserFcts, FreeDyn) 
        Ctrl.update_vecC(tRight/dataOpt.final_time)
        FreeDyn.update_dForce_dFDparam()
        adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.SysMat.dForce_dFDparam
        np.multiply.outer(adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_Phi_buff[idx_buff])
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
        tLeft = FreeDyn.API.t
        deltaT = tRight - tLeft
        self.BDF.BDForder1_singleStep_Phi(FreeDyn, deltaT)
        Ctrl.update_vecC(tLeft/dataOpt.final_time)
        FreeDyn.update_dForce_dFDparam()
        adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.SysMat.dForce_dFDparam
        np.multiply.outer(adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_Phi_buff[idx_buff])
        dPhidu = deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
        
        """ BDF order 2 """      
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            tRight = tLeft
            FreeDyn.fetch_and_update_states_at_index(i)
            tLeft = FreeDyn.API.t
            deltaT = tRight - tLeft
            self.BDF.BDForder2_singleStep_Phi(FreeDyn, deltaT)
            Ctrl.update_vecC(tLeft/dataOpt.final_time)
            FreeDyn.update_dForce_dFDparam()
            adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.SysMat.dForce_dFDparam
            np.multiply.outer(adjP_fdu, Ctrl.buff_vecC, out = self.adjGrad_Phi_buff[idx_buff])
            dPhidu += deltaT * (self.adjGrad_Phi_buff_view0 + self.adjGrad_Phi_buff_view1)
            
        dPhidu *= 0.5
                
        return dPhidu
# -----------------------------------------------------------------------------        

    # def adjGrad_Phi(self, dataOpt, UserFcts, FreeDyn, Ctrl, z):
        
    #     """ Gradient of the cost functioncal J w.r.t. uDach 
    #         Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        
    #     """ t = t_f / init BDF routine """
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-1)
    #     t_i = FreeDyn.API.t
    #     self.BC_FDOP.get_consistent_BC_Phi(self.BDF, UserFcts, FreeDyn)
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #     np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff)
        
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-2)
    #     t_i_min = FreeDyn.API.t
        
    #     dPhidu = (t_i - t_i_min) * self.adjGrad_Phi_buff_view
        
        
    #     """ BDF order 1 """
    #     t_i_plus = t_i
    #     t_i = t_i_min
    #     BDF_delta_t = t_i_plus - t_i
    #     self.BDF.BDForder1_singleStep_Phi(FreeDyn, BDF_delta_t)        
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #     np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff)
        
    #     FreeDyn.fetch_and_update_states_at_index(FreeDyn.num_time_steps-3)
    #     t_i_min = FreeDyn.API.t
    #     dPhidu += (t_i_plus - t_i_min) * self.adjGrad_Phi_buff_view
        
                
    #     """ BDF order 2 """        
    #     for i in range(FreeDyn.num_time_steps-3, 0, -1):
    #         t_i_plus = t_i
    #         t_i = t_i_min
    #         BDF_delta_t = t_i_plus - t_i
    #         self.BDF.BDForder2_singleStep_Phi(FreeDyn, BDF_delta_t)
    #         FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #         adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
    #         vec_C = Ctrl.get_vec_c(t_i/dataOpt.final_time)
    #         np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff)
            
    #         FreeDyn.fetch_and_update_states_at_index(i-1)
    #         t_i_min = FreeDyn.API.t
    #         dPhidu += (t_i_plus - t_i_min) * self.adjGrad_Phi_buff_view
            
        
    #     BDF_delta_t = t_i - t_i_min
    #     self.BDF.BDForder2_singleStep_Phi(FreeDyn, BDF_delta_t)
    #     FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
    #     adjP_fdu = self.BDF.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
    #     vec_C = Ctrl.get_vec_c(t_i_min/dataOpt.final_time)
    #     np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_buff)  
    #     dPhidu += (t_i - t_i_min) * self.adjGrad_Phi_buff_view         
            
    #     dPhidu *= 0.5 
                
    #     return dPhidu