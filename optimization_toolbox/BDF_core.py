import numpy as np 

class BDF():
    
    @classmethod
    def create(cls, FreeDyn, num_xF):
        if FreeDyn.MBS_modeMAT_sparse:
            from .BDF_sparse_physical_time import SparseBDF
            return SparseBDF(FreeDyn, num_xF)
        else:
            from .BDF_dense_physical_time import DenseBDF
            return DenseBDF(FreeDyn, num_xF)
    
    def __init__(self, FreeDyn, num_xF):
        
        # Create buffers for fast access
        self.init_buff_indices()
        self.allocate_buffs_BDF(FreeDyn, num_xF)       
# -----------------------------------------------------------------------------
        
    def init_buff_indices(self):
        self.buff_idx1 = 0
        self.buff_idx2 = 1 - self.buff_idx1
        
    def shift_buff_indices(self):
        self.buff_idx1 = self.buff_idx2
        self.buff_idx2 = 1 - self.buff_idx1
            
# =============================================================================
# Allocate Buffers
# =============================================================================
    
    def allocate_buffs_BDF(self, FreeDyn, num_xF):
        
        # Run memory layout dependent setup (add. buffers, coeff. matrix of adjoint system)
        self.init_memory_layout_dependent(FreeDyn, num_xF)
        
        # Buffers independent of memory layout
        self.adjP_J = np.zeros(FreeDyn.nDof) 
        self.buff_adjW_J = np.zeros((2, FreeDyn.nDof))
        self.buff_M_x_AdjP_J = np.zeros((2, FreeDyn.nDof))
        if num_xF > 0:
            self.adjP_Phi = np.zeros((FreeDyn.nDof, num_xF)) 
            self.buff_adjW_Phi = np.zeros((2, FreeDyn.nDof, num_xF))
            self.buff_M_x_AdjP_Phi = np.zeros((2, FreeDyn.nDof, num_xF))
        
        # Buffers of time diffs for etas        
        self.BDF_diff_tau = np.zeros(2)
        
        # Buffers of eta BDF order 1
        self.BDF1_eta0 = 0.0
        self.BDF1_eta1 = 0.0
        
        # Buffers of eta BDF order 2
        self.BDF2_eta0 = 0.0
        self.BDF2_eta1 = 0.0
        self.BDF2_eta2 = 0.0

# =============================================================================
# Set initial values for the BDF routines
# =============================================================================

    def set_adj_BC_J(self, init_adjP, init_adjW):
        self.init_buff_indices()
        self.adjP_J[:] = init_adjP
        self.buff_adjW_J[:] = init_adjW

    def set_adj_BC_Phi(self, init_adjP, init_adjW):
        self.init_buff_indices()
        self.adjP_Phi[:] = init_adjP
        self.buff_adjW_Phi[:] = init_adjW
        
# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================
    
    def order1_coeffs_eta(self, deltaT):
        
        self.BDF_diff_tau[self.buff_idx1] = deltaT
        
        self.BDF1_eta0 = 1 / deltaT
        self.BDF1_eta1 = -self.BDF1_eta0
# -----------------------------------------------------------------------------        
        
    def BDForder1_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        self.shared_pre_steps_J(UserFcts, FreeDyn, z)                           # memory independent
        self.order1_coeffs_eta(deltaT)
        self.order1_post_steps_J(UserFcts, FreeDyn)                             # memory dependent

    def BDForder1_singleStep_Phi(self, FreeDyn, deltaT):
        self.shared_pre_steps_Phi(FreeDyn)                                      # memory independent
        self.order1_coeffs_eta(deltaT)
        self.order1_post_steps_Phi(FreeDyn)                                     # memory dependent

# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def order2_coeffs_eta(self, diff_01):
        
        self.BDF_diff_tau[self.buff_idx1] = diff_01                             # s_{n} - s_{n-1}
        diff_12 = self.BDF_diff_tau[self.buff_idx2]                             # s_{n-1} - s_{n-2}
        diff_02 = diff_01 + diff_12                                             # s_{n} - s_{n-2}

        self.BDF2_eta0 = (diff_01 + diff_02) / (diff_01 * diff_02)              # eta_0
        self.BDF2_eta1 = -diff_02 / (diff_01 * diff_12)                         # eta_1
        self.BDF2_eta2 = diff_01 / (diff_02 * diff_12)                          # eta_2 
# -----------------------------------------------------------------------------        
        
    def BDForder2_singleStep_J(self, UserFcts, FreeDyn, z, deltaT):
        self.shared_pre_steps_J(UserFcts, FreeDyn, z)                           # memory independent
        self.order2_coeffs_eta(deltaT)
        self.order2_post_steps_J(UserFcts, FreeDyn)                             # memory dependent

    def BDForder2_singleStep_Phi(self, FreeDyn, deltaT):
        self.shared_pre_steps_Phi(FreeDyn)                                      # memory independent
        self.order2_coeffs_eta(deltaT)
        self.order2_post_steps_Phi(FreeDyn)                                     # memory dependent
        
# =============================================================================
# BDF Routine: Shared pre steps - memory independent
# =============================================================================

    def shared_pre_steps_J(self, UserFcts, FreeDyn, z):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_J[self.buff_idx1, :] = FreeDyn.SysMat.M @ self.adjP_J

        # Updates user functions to time step s_n
        UserFcts.get_lagrangian_dq(FreeDyn, z)
        UserFcts.get_lagrangian_dv(FreeDyn, z)
        
        # update_sys_mat_for_BDF() is carried out within _dense/_spare
# ----------------------------------------------------------------------------- 

    def shared_pre_steps_Phi(self, FreeDyn):
        
        # Compute M(s_n-1) * adjP(s_n-1) - must before update_MBS_SysMat, because s_n-1 is needed!
        self.buff_M_x_AdjP_Phi[self.buff_idx1, :, :] = FreeDyn.SysMat.M @ self.adjP_Phi
        
        # update_sys_mat_for_BDF() is carried out within _dense/_spare