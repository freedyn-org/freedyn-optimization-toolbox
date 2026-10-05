import numpy as np

from core.consistent_boundary_conditions import BC_FDOP
from core.BDF_physicalTime import BDF

class adjGrads_TOCP(BC_FDOP, BDF):
    
    def __init__(self, dataOpt, FreeDyn, Ctrl):
        
        BC_FDOP.__init__(self, dataOpt, FreeDyn)
        BDF.__init__(self, dataOpt, FreeDyn)
        
        self.adjGrad_J_uDach_buff = np.zeros((2, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
        self.adjGrad_J_uDach_buff_view0 = self.adjGrad_J_uDach_buff[0].reshape(-1)
        self.adjGrad_J_uDach_buff_view1 = self.adjGrad_J_uDach_buff[1].reshape(-1)
        
        if dataOpt.num_xF > 0: 
            ctrl_all = Ctrl.num_ctrls * Ctrl.num_grid_nodes
            self.adjGrad_Phi_uDach_buff = np.zeros((2, dataOpt.num_xF, Ctrl.num_ctrls, Ctrl.num_grid_nodes))
            self.adjGrad_Phi_uDach_buff_view0 = self.adjGrad_Phi_uDach_buff[0].reshape(dataOpt.num_xF, ctrl_all)
            self.adjGrad_Phi_uDach_buff_view1 = self.adjGrad_Phi_uDach_buff[1].reshape(dataOpt.num_xF, ctrl_all)
# -----------------------------------------------------------------------------
    
    def adjGrad_updates(self, FreeDyn, idx):
        
        FreeDyn.API.fetch_states_at_index(idx)
        FreeDyn.API.update_state_at_index(idx)
        FreeDyn.API.update_jacobian()
        FreeDyn.buffer_MBS_dForce_dFDparam.update_from_dll()
        
        return FreeDyn.API.t
# -----------------------------------------------------------------------------
    
    def adjGrad_J(self, dataOpt, user_Fcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the cost functioncal J w.r.t. uDach + final time t_f 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        gradJ = np.zeros(dataOpt.num_opt_vars)
        Ctrl.vec_c_dtF_invariant_OptIt(dataOpt.final_time)
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-1)
        self.get_consistent_BC_J(FreeDyn)
        user_Fcts.get_Lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
        dLdu_adjP_fdu = user_Fcts.dLdu + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
        vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tRight/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Right = tRight * (dLdu_adjP_fdu @ dCdtau_times_uDach)
        L_tF = user_Fcts.get_Lagrangian(dataOpt, FreeDyn, Ctrl, z) # is added at the end, otherwise multiplied by 0.5
        
        """ BDF order 1 """
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_J(user_Fcts, FreeDyn, z, deltaT)        
        user_Fcts.get_Lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
        dLdu_adjP_fdu = user_Fcts.dLdu + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
        vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tLeft/dataOpt.final_time)
        np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_times_uDach)
        gradJ[0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradJ[1:] += deltaT * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
                
        """ BDF order 2 """        
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
            tLeft = self.adjGrad_updates(FreeDyn, i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_J(user_Fcts, FreeDyn, z, deltaT)
            user_Fcts.get_Lagrangian_du(dataOpt, FreeDyn, Ctrl, z)
            dLdu_adjP_fdu = user_Fcts.dLdu + self.get_adjVar_p_J().T @ FreeDyn.dForce_dFDparam
            vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tLeft/dataOpt.final_time)
            np.outer(dLdu_adjP_fdu, vec_C, out = self.adjGrad_J_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (dLdu_adjP_fdu @ dCdtau_times_uDach)
            gradJ[0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
            gradJ[1:] += deltaT * (self.adjGrad_J_uDach_buff_view0 + self.adjGrad_J_uDach_buff_view1)
        
        gradJ *= 0.5
        gradJ[0] += L_tF  # is added here, otherwise multiplied by 0.5

        return gradJ
# -----------------------------------------------------------------------------    
    
    def adjGrad_Phi(self, dataOpt, user_Fcts, FreeDyn, Ctrl, z):
        
        """ Gradient of the final constraints Phi w.r.t. uDach + final time t_f 
            Numerical integration by the trapezoidal rule: use t_i , t_i+1 """
        gradPhi = np.zeros([dataOpt.num_xF, dataOpt.num_opt_vars])
        Ctrl.vec_c_dtF_invariant_OptIt(dataOpt.final_time)
        
        """ t = t_f / init BDF routine """
        idx_buff = 0
        tRight = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-1)
        self.get_consistent_BC_Phi(user_Fcts, FreeDyn) # this updates dPhidq and dPhidv
        dPhidt_tF = user_Fcts.dPhidq @ FreeDyn.API.Qd[:, 0] + user_Fcts.dPhidv @ FreeDyn.API.Qdd[:, 0]  # is added at the end, otherwise multiplied by 0.5y()
        adjP_fdu = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tRight/dataOpt.final_time)
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])        
        integrand_tF_Right = tRight * (adjP_fdu @ dCdtau_times_uDach)
        
        """ BDF order 1 """ 
        idx_buff = 1 - idx_buff
        tLeft = self.adjGrad_updates(FreeDyn, FreeDyn.num_time_steps-2)
        deltaT = tRight - tLeft
        self.BDForder1_singleStep_Phi(FreeDyn, deltaT)
        adjP_fdu = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
        vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tLeft/dataOpt.final_time)
        np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
        integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_times_uDach)
        gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
        gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)   
        
        """ BDF order 2 """      
        for i in range(FreeDyn.num_time_steps-3, -1, -1):
            idx_buff = 1 - idx_buff
            integrand_tF_Right = integrand_tF_Left
            tRight = tLeft
            tLeft = self.adjGrad_updates(FreeDyn, i)
            deltaT = tRight - tLeft
            self.BDForder2_singleStep_Phi(FreeDyn, deltaT)
            adjP_fdu = self.get_adjVar_P_Phi().T @ FreeDyn.dForce_dFDparam
            vec_C, dCdtau_times_uDach = Ctrl.get_vec_c_AND_dCdtau_times_uDach(tLeft/dataOpt.final_time)
            np.multiply(adjP_fdu[:,:,np.newaxis], vec_C, out = self.adjGrad_Phi_uDach_buff[idx_buff])
            integrand_tF_Left = tLeft * (adjP_fdu @ dCdtau_times_uDach)
            gradPhi[:,0] -= deltaT * (integrand_tF_Left + integrand_tF_Right)
            gradPhi[:,1:] += deltaT * (self.adjGrad_Phi_uDach_buff_view0 + self.adjGrad_Phi_uDach_buff_view1)
        
        gradPhi *= 0.5
        gradPhi[:,0] += dPhidt_tF # is added here, otherwise multiplied by 0.5
                
        return gradPhi
# -----------------------------------------------------------------------------    