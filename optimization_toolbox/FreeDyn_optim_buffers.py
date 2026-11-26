import numpy as np
from ctypes import c_int
import freedyn as fd

class FreeDynBuffers():
    
    @classmethod
    def create(cls, memory_layout, name_dForce_dparam):
        if memory_layout:
            return FreeDynBuffersSparse(memory_layout, name_dForce_dparam)
        else:
            return FreeDynBuffersDense(memory_layout, name_dForce_dparam)

# -----------------------------------------------------------------------------

    def __init__(self, memory_layout, name_dForce_dparam):
        
        self.init_dForce_dFDparam(name_dForce_dparam)
        self.init_sys_mat_slots()
        self.init_sys_mat_slots_for_adj_BC(memory_layout)
# -----------------------------------------------------------------------------
            
    def init_dForce_dFDparam(self, name_dForce_dparam):
        self.buffer_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(name_dForce_dparam)
        self.dForce_dFDparam = self.buffer_dForce_dFDparam.data
# -----------------------------------------------------------------------------        
        
    def init_sys_mat_slots_for_adj_BC(self, memory_layout):
        attr_name = 'sp_mat' if memory_layout else 'dense_mat'
        
        scale = np.array([1.0, 1.0, 1.0])
        row_pos = np.array([0, 0, 1], dtype=c_int)
        col_pos = np.array([0, 1, 0], dtype=c_int)
        id_adj_w = np.array([112, 201, 301], dtype=c_int)
        id_adj_p = np.array([101, 201, 301], dtype=c_int)
        coeff_mat_BC_adjW_idx = fd.analysis.create_matrix(id_adj_w, row_pos, col_pos, scale)
        coeff_mat_BC_adjP_idx = fd.analysis.create_matrix(id_adj_p, row_pos, col_pos, scale)
        self.slot_coeff_mat_BC_adjW = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjW_idx, memory_layout)
        self.slot_coeff_mat_BC_adjP = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjP_idx, memory_layout)
        self.coeff_mat_BC_adjW = getattr(self.slot_coeff_mat_BC_adjW, attr_name) 
        self.coeff_mat_BC_adjP = getattr(self.slot_coeff_mat_BC_adjP, attr_name) 
        
# ----------------------------------------------------------------------------- 

    def update_dForce_dFDparam(self):
        self.buffer_dForce_dFDparam.update_from_dll()
# ----------------------------------------------------------------------------- 
            
    def update_sys_mat_for_consistent_BC_J(self, API):
        API.update_jacobian()
        self.slot_M.update_from_dll()
        self.slot_M.apply_to_cached_matrix()
# -----------------------------------------------------------------------------

    def update_sys_mat_for_consistent_BC_Phi(self, API):
        API.update_jacobian()
        self.slot_M.update_from_dll()
        self.slot_M.apply_to_cached_matrix()
        self.slot_CqvDq.update_from_dll()
        self.slot_CqvDq.apply_to_cached_matrix()
        
        self.slot_coeff_mat_BC_adjW.update_from_dll()
        self.slot_coeff_mat_BC_adjW.apply_to_cached_matrix()
        self.slot_coeff_mat_BC_adjP.update_from_dll()
        self.slot_coeff_mat_BC_adjP.apply_to_cached_matrix()
# -----------------------------------------------------------------------------


class FreeDynBuffersDense(FreeDynBuffers):

    def init_sys_mat_slots(self):
        
        # Row/Column position and scaling value of single matrix
        pos_mat = np.array([0], dtype=c_int)
        scale_mat = np.array([1.0])
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_M = fd.ModelRelatedMatrixBuffer(M_idx, False)
        self.M = getattr(self.slot_M, 'dense_mat') 
        
        # Constraint Jacobian Cq
        id_Cq = np.array([301], dtype=c_int)
        Cq_idx = fd.analysis.create_matrix(id_Cq, pos_mat, pos_mat, scale_mat)
        self.slot_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, False)
        self.Cq = getattr(self.slot_Cq, 'dense_mat') 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, False)
        self.CqvDq = getattr(self.slot_CqvDq, 'dense_mat') 
        
        # fv
        id_fv = np.array([109], dtype=c_int)
        fv_idx = fd.analysis.create_matrix(id_fv, pos_mat, pos_mat, scale_mat)
        self.slot_fv = fd.ModelRelatedMatrixBuffer(fv_idx, False)
        self.fv = getattr(self.slot_fv, 'dense_mat') 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        id_matG = np.array([108, 110, 105, 102], dtype=c_int)
        pos_matG = np.array([0, 0, 0, 0], dtype=c_int)
        scale_matG = np.array([1.0, -1.0, -1.0, -1.0])
        G_idx = fd.analysis.create_matrix(id_matG, pos_matG, pos_matG, scale_matG)
        self.slot_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, False)
        self.G_tr = getattr(self.slot_G_tr, 'dense_mat')
# ----------------------------------------------------------------------------- 

    def update_sys_mat_for_BDF(self, API):
        API.update_jacobian()
        self.slot_M.update_from_dll()
        self.slot_M.apply_to_cached_matrix()
        self.slot_Cq.update_from_dll()
        self.slot_Cq.apply_to_cached_matrix()
        self.slot_CqvDq.update_from_dll()
        self.slot_CqvDq.apply_to_cached_matrix()
        self.slot_fv.update_from_dll()
        self.slot_fv.apply_to_cached_matrix()
        self.slot_G_tr.update_from_dll()   
        self.slot_G_tr.apply_to_cached_matrix()
# -----------------------------------------------------------------------------     


class FreeDynBuffersSparse(FreeDynBuffers):
    
    def init_sys_mat_slots(self):
        
        # Row/Column position and scaling value of single matrix
        pos_mat = np.array([0], dtype=c_int)
        scale_mat = np.array([1.0])
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_M = fd.ModelRelatedMatrixBuffer(M_idx, True)
        self.M = getattr(self.slot_M, 'sp_mat') 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, True)
        self.CqvDq = getattr(self.slot_CqvDq, 'sp_mat') 

        # Coeff. matrix of adjoint system
        self.BDF_mat_eta_pos = np.array([0,8])
        self.BDF_mat_eta = np.array([1.0, 1.0]) # dummy values - are replaced
        
        id_BDF_mat = np.array([
            # Row 1: eta0*I | -G | -(Cq v)_q^Tr | Cq^Tr
            112, 128, 130, 125, 122, 202, 201,
            # Row 2: -I | eta0*M - fv^Tr | -Cq^Tr | 0
            112, 101, 129, 201,
            # Row 3: 0 | Cq | 0 | 0
            301,
            # Row 4: Cq | 0 | 0 | 0
            301], dtype=c_int)
        row_BDF_mat = np.array([0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1,
                                2,
                                3], dtype=c_int)
        col_BDF_mat = np.array([0, 1, 1, 1, 1, 2, 3,
                                0, 1, 1, 2,
                                1,
                                0], dtype=c_int)
        BDF_mat_scale = np.array([self.BDF_mat_eta[0], -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
                                  -1.0, self.BDF_mat_eta[1], -1.0, -1.0,
                                  1.0,
                                  1.0])
        BDF_mat_idx = fd.analysis.create_matrix(id_BDF_mat, row_BDF_mat, col_BDF_mat, BDF_mat_scale)
        self.slot_BDF_mat = fd.ModelRelatedMatrixBuffer(BDF_mat_idx, True)
        self.BDF_mat = getattr(self.slot_BDF_mat, 'sp_mat')
        
        self.slot_BDF_mat.create_internal_scaling_values(BDF_mat_scale)
# -----------------------------------------------------------------------------

    def update_BDF_mat_eta(self, eta0):
        self.BDF_mat_eta.fill(eta0)
        self.slot_BDF_mat.set_scaling_from_internal_values(self.BDF_mat_eta_pos, self.BDF_mat_eta)
# ----------------------------------------------------------------------------- 

    def update_sys_mat_for_BDF(self, API):
        API.update_jacobian()
        self.slot_BDF_mat.update_from_dll()
        self.slot_M.update_from_dll()
        
        self.slot_M.apply_to_cached_matrix()        
# ----------------------------------------------------------------------------- 