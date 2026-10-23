import numpy as np
from ctypes import c_int
import freedyn as fd


class FreeDynBuffers():
    
    def __init__(self, 
                 memory_layout, name_dForce_dparam):
        
        self.init_sys_mat_slots(memory_layout)
        self.init_dForce_dFDparam(name_dForce_dparam)
# -----------------------------------------------------------------------------         
        
    def init_dForce_dFDparam(self, name_dForce_dparam):
        
        self.buffer_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(name_dForce_dparam)
        self.dForce_dFDparam = self.buffer_dForce_dFDparam.data
# ----------------------------------------------------------------------------- 

    def update_dForce_dFDparam(self):
        self.buffer_dForce_dFDparam.update_from_dll()
    
    def init_sys_mat_slots(self, memory_layout):
        
        # Row/Column position and scaling value of single matrix
        pos_mat = np.array([0], dtype=c_int)
        scale_mat = np.array([1.0])
        
        # Set up the memory layout either as sparse or dense
        attr_name = 'sp_mat' if memory_layout else 'dense_mat'
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_M = fd.ModelRelatedMatrixBuffer(M_idx, memory_layout)
        self.M = getattr(self.slot_M, attr_name) 
        
        # Constraint Jacobian Cq
        id_Cq = np.array([301], dtype=c_int)
        Cq_idx = fd.analysis.create_matrix(id_Cq, pos_mat, pos_mat, scale_mat)
        self.slot_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, memory_layout)
        self.Cq = getattr(self.slot_Cq, attr_name) 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, memory_layout)
        self.CqvDq = getattr(self.slot_CqvDq, attr_name) 
        
        # fv
        id_fv = np.array([109], dtype=c_int)
        fv_idx = fd.analysis.create_matrix(id_fv, pos_mat, pos_mat, scale_mat)
        self.slot_fv = fd.ModelRelatedMatrixBuffer(fv_idx, memory_layout)
        self.fv = getattr(self.slot_fv, attr_name) 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        id_matG = np.array([108, 110, 105, 102], dtype=c_int)
        pos_matG = np.array([0, 0, 0, 0], dtype=c_int)
        scale_matG = np.array([1.0, -1.0, -1.0, -1.0])
        G_idx = fd.analysis.create_matrix(id_matG, pos_matG, pos_matG, scale_matG)
        self.slot_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, memory_layout)
        self.G_tr = getattr(self.slot_G_tr, attr_name)    
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

    def update_nnz_dll_sys_mat_for_BDF(self, API):
        API.update_jacobian()
        self.slot_M.update_from_dll()
        self.slot_Cq.update_from_dll()
        self.slot_CqvDq.update_from_dll()
        self.slot_fv.update_from_dll()
        self.slot_G_tr.update_from_dll()   
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
        self.slot_Cq.update_from_dll()
        self.slot_Cq.apply_to_cached_matrix()
        self.slot_CqvDq.update_from_dll()
        self.slot_CqvDq.apply_to_cached_matrix()
# -----------------------------------------------------------------------------