import numpy as np

class CoreBDF():
    
    def __init__(self, dataOpt, FreeDyn, nBDFsys, partition_solVec_1, partition_solVec_2):
        
        # Create buffers for fast access
        self.BDF_idx_buff = 0
        num_buffs = 2   # do not change - buffer layout changes require fixes in many places

        # Allocate memory for the coeff. Matrix of the adj Sys      
        self.allocate_buffs_BDF(num_buffs, FreeDyn.nDof, dataOpt.num_xF, nBDFsys, partition_solVec_1, partition_solVec_2)
        
        self.init_order_one_BDF()
        self.init_order_two_BDF()
        
        print('class BDF initialized')
        
# =============================================================================
# Return values of adjVar p at BDF time idx s_n from the buffer
# =============================================================================

    def get_adjVar_p_J(self):
        return self.adjP_J_buff[self.BDF_idx_buff, :]

    def get_adjVar_P_Phi(self):
        return self.adjP_Phi_buff[self.BDF_idx_buff,:,:]
            
# =============================================================================
# Allocate Buffers
# =============================================================================
    
    def allocate_buffs_BDF(self, num_buffs, num_q, num_xF, dim_sys, partition_1, partition_2):
        
        self.BDF_diff_tau = np.zeros(num_buffs)
        
        self.adjW_J_buff = np.empty((num_buffs, num_q))
        self.adjP_J_buff = np.empty((num_buffs, num_q))
        self.BDF_J_buff_M_times_p = np.zeros((num_buffs, num_q))
        
        self.BDF_solVec_J = np.zeros(dim_sys)
        self.BDF_solVec_J_view_first = self.BDF_solVec_J[:partition_1]
        self.BDF_solVec_J_view_second = self.BDF_solVec_J[partition_1:partition_2]
        
        if num_xF > 0:
            self.adjW_Phi_buff = np.empty((num_buffs, num_q, num_xF))
            self.adjP_Phi_buff = np.empty((num_buffs, num_q, num_xF)) 
            self.BDF_Phi_buff_M_times_P = np.zeros((num_buffs, num_q, num_xF))
            
            self.BDF_solVec_Phi = np.zeros((dim_sys, num_xF))
            self.BDF_solVec_Phi_view_first = self.BDF_solVec_Phi[:partition_1, :]
            self.BDF_solVec_Phi_view_second = self.BDF_solVec_Phi[partition_1:partition_2,:]           
# ----------------------------------------------------------------------------- 
            
    def allocate_buffs_BDF_add_dense(self, num_q, num_xF):
        
        self.BDF_J_eta_times_adjW = np.empty(num_q)
        if num_xF > 0:
            self.BDF_Phi_eta_times_adjW = np.empty((num_q, num_xF))        
        
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