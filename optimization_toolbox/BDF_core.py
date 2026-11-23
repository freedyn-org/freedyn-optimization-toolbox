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
        self.BDF_idx_buff = 0
        
        self.init_memory_layout_dependent(FreeDyn, num_xF)
        
        # Allocate memory for the coeff. Matrix of the adj Sys 
        nBDFsys = self.init_coeff_mat_adj_sys(FreeDyn)
        
        self.allocate_buffs_BDF(FreeDyn.nDof, num_xF, nBDFsys)
        
        self.init_order_one_BDF()
        self.init_order_two_BDF()
        
        # print('class BDF initialized')
        
# =============================================================================
# Return values of adjVar p at BDF time idx s_n from the buffer
# =============================================================================

    def get_adjVar_p_J(self):
        return self.buff_adjP_J[self.BDF_idx_buff, :]

    def get_adjVar_P_Phi(self):
        return self.buff_adjP_Phi[self.BDF_idx_buff,:,:]
            
# =============================================================================
# Allocate Buffers
# =============================================================================
    
    def allocate_buffs_BDF(self, num_q, num_xF, dim_sys):
        
        self.BDF_diff_tau = np.zeros(2)
        
        self.buff_adjW_J = np.empty((2, num_q))
        self.buff_adjP_J = np.empty((2, num_q))
        self.buff_M_x_AdjP_J = np.zeros((2, num_q))
        self.buff_solVec_J = np.zeros(dim_sys)
        
        if num_xF > 0:
            self.buff_adjW_Phi = np.empty((2, num_q, num_xF))
            self.buff_adjP_Phi = np.empty((2, num_q, num_xF)) 
            self.buff_M_x_AdjP_Phi = np.zeros((2, num_q, num_xF))
            self.buff_solVec_Phi = np.zeros((dim_sys, num_xF))    
        
# =============================================================================
# BDF Routine: Int. order 1
# =============================================================================
    
    def init_order_one_BDF(self):
        self.BDF1_eta0_inv = 0.0
        self.BDF1_eta0 = 0.0
        self.BDF1_eta1 = 0.0
# -----------------------------------------------------------------------------
    
    def get_BDForder1_coeffs_eta(self, idx1, deltaT):
        
        self.BDF_diff_tau[idx1] = deltaT

        self.BDF1_eta0_inv = deltaT
        self.BDF1_eta0 = 1 / deltaT
        self.BDF1_eta1 = -self.BDF1_eta0 

# =============================================================================
# BDF Routine: Int. order 2
# =============================================================================

    def init_order_two_BDF(self):
        
        self.BDF2_eta0_inv = 0.0
        self.BDF2_eta0 = 0.0
        self.BDF2_eta1 = 0.0
        self.BDF2_eta2 = 0.0        
# -----------------------------------------------------------------------------

    def get_BDForder2_coeffs_eta(self, idx1, idx2, diff_01):
        
        self.BDF_diff_tau[idx1] = diff_01                                       # s_{n} - s_{n-1}
        diff_12 = self.BDF_diff_tau[idx2]                                       # s_{n-1} - s_{n-2}
        diff_02 = diff_01 + diff_12                                             # s_{n} - s_{n-2}

        self.BDF2_eta0_inv = (diff_01 * diff_02) / (diff_01 + diff_02)          # inv(eta0)
        
        self.BDF2_eta0 = 1 / self.BDF2_eta0_inv                                 # eta_0
        self.BDF2_eta1 = -diff_02 / (diff_01 * diff_12)                         # eta_1
        self.BDF2_eta2 = diff_01 / (diff_02 * diff_12)                          # eta_2 
        
# -----------------------------------------------------------------------------
