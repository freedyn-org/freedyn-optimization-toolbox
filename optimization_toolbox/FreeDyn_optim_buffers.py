import numpy as np
from ctypes import c_int
import freedyn as fd


class FreeDynBuffers():
    
    def __init__(self):
        
        self.init_MBS_sysMat_slots()
        self.init_MBS_dForce_dFDparam()
# -----------------------------------------------------------------------------         
        
    def init_MBS_dForce_dFDparam(self):
        
        self.buffer_MBS_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(self.fds_dForce_dparam_name)
        self.dForce_dFDparam = self.buffer_MBS_dForce_dFDparam.data
# ----------------------------------------------------------------------------- 
    
    def init_MBS_sysMat_slots(self):
        
        # Row/Column position and scaling value of single matrix
        pos_mat = np.array([0], dtype=c_int)
        scale_mat = np.array([1.0])
        
        # Set up the memory layout either as sparse or dense
        attr_name = 'sp_mat' if self.MBS_modeMAT_sparse else 'dense_mat'
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_M = fd.ModelRelatedMatrixBuffer(M_idx, self.MBS_modeMAT_sparse)
        self.MBS_M = getattr(self.slot_MBS_M, attr_name) 
        
        # Constraint Jacobian Cq
        id_Cq = np.array([301], dtype=c_int)
        Cq_idx = fd.analysis.create_matrix(id_Cq, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, self.MBS_modeMAT_sparse)
        self.MBS_Cq = getattr(self.slot_MBS_Cq, attr_name) 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, self.MBS_modeMAT_sparse)
        self.MBS_CqvDq = getattr(self.slot_MBS_CqvDq, attr_name) 
        
        # fv
        id_fv = np.array([109], dtype=c_int)
        fv_idx = fd.analysis.create_matrix(id_fv, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_fv = fd.ModelRelatedMatrixBuffer(fv_idx, self.MBS_modeMAT_sparse)
        self.MBS_fv = getattr(self.slot_MBS_fv, attr_name) 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        id_matG = np.array([108, 110, 105, 102], dtype=c_int)
        pos_matG = np.array([0, 0, 0, 0], dtype=c_int)
        scale_matG = np.array([1.0, -1.0, -1.0, -1.0])
        G_idx = fd.analysis.create_matrix(id_matG, pos_matG, pos_matG, scale_matG)
        self.slot_MBS_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, self.MBS_modeMAT_sparse)
        self.MBS_G_tr = getattr(self.slot_MBS_G_tr, attr_name)    
# ----------------------------------------------------------------------------- 

    def update_sys_mat_for_BDF(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_fv.apply_to_cached_matrix()
        self.slot_MBS_G_tr.update_from_dll()   
        self.slot_MBS_G_tr.apply_to_cached_matrix()
# ----------------------------------------------------------------------------- 

    def update_nnz_dll_sys_mat_for_BDF(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_G_tr.update_from_dll()   
# -----------------------------------------------------------------------------
    
    def update_sys_mat_for_consistent_BC_J(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
# -----------------------------------------------------------------------------

    def update_sys_mat_for_consistent_BC_Phi(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
# -----------------------------------------------------------------------------