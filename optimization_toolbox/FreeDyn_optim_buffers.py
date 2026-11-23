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
        
        # if memory_layout:
        #     self.init_BDF_sparse_coeff_mat_slot()

# -----------------------------------------------------------------------------
            
    def init_dForce_dFDparam(self, name_dForce_dparam):
        self.buffer_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(name_dForce_dparam)
        self.dForce_dFDparam = self.buffer_dForce_dFDparam.data
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
        
        # Coeff matrices for adjoint variables - BC
        scale_coeff_mat_BC_adj = np.array([1.0, 1.0, 1.0])
        row_coeff_mat_BC_adj = np.array([0, 0, 1], dtype=c_int)
        col_coeff_mat_BC_adj = np.array([0, 1, 0], dtype=c_int)
        id_coeff_mat_BC_adjW = np.array([112, 201, 301], dtype=c_int)
        id_coeff_mat_BC_adjP = np.array([101, 201, 301], dtype=c_int)
        coeff_mat_BC_adjW_idx = fd.analysis.create_matrix(id_coeff_mat_BC_adjW, row_coeff_mat_BC_adj, col_coeff_mat_BC_adj, scale_coeff_mat_BC_adj)
        coeff_mat_BC_adjP_idx = fd.analysis.create_matrix(id_coeff_mat_BC_adjP, row_coeff_mat_BC_adj, col_coeff_mat_BC_adj, scale_coeff_mat_BC_adj)
        self.slot_coeff_mat_BC_adjW = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjW_idx, False)
        self.slot_coeff_mat_BC_adjP = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjP_idx, False)
        self.coeff_mat_BC_adjW = getattr(self.slot_coeff_mat_BC_adjW, 'dense_mat') 
        self.coeff_mat_BC_adjP = getattr(self.slot_coeff_mat_BC_adjP, 'dense_mat') 
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
        
        # Identity matrix I
        id_I = np.array([112], dtype=c_int)
        I_idx = fd.analysis.create_matrix(id_I, pos_mat, pos_mat, scale_mat)
        self.slot_I = fd.ModelRelatedMatrixBuffer(I_idx, True)
        self.I = getattr(self.slot_I, 'sp_mat') 
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_M = fd.ModelRelatedMatrixBuffer(M_idx, True)
        self.M = getattr(self.slot_M, 'sp_mat') 
        
        # Constraint Jacobian Cq
        id_Cq = np.array([301], dtype=c_int)
        Cq_idx = fd.analysis.create_matrix(id_Cq, pos_mat, pos_mat, scale_mat)
        self.slot_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, True)
        self.Cq = getattr(self.slot_Cq, 'sp_mat') 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, True)
        self.CqvDq = getattr(self.slot_CqvDq, 'sp_mat') 
        
        # M - fv
        id_M_fv = np.array([101,129], dtype=c_int)
        pos_M_fv = np.array([0, 0], dtype=c_int)
        scale_M_fv = np.array([1.0, -1.0])
        M_fv_idx = fd.analysis.create_matrix(id_M_fv, pos_M_fv, pos_M_fv, scale_M_fv)
        self.slot_M_fv = fd.ModelRelatedMatrixBuffer(M_fv_idx, True)
        self.M_fv = getattr(self.slot_M_fv, 'sp_mat') 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        id_matG = np.array([108, 110, 105, 102], dtype=c_int)
        pos_matG = np.array([0, 0, 0, 0], dtype=c_int)
        scale_matG = np.array([1.0, -1.0, -1.0, -1.0])
        G_idx = fd.analysis.create_matrix(id_matG, pos_matG, pos_matG, scale_matG)
        self.slot_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, True)
        self.G_tr = getattr(self.slot_G_tr, 'sp_mat')
        
        # Coeff matrices for adjoint variables - BC
        scale_coeff_mat_BC_adj = np.array([1.0, 1.0, 1.0])
        row_coeff_mat_BC_adj = np.array([0, 0, 1], dtype=c_int)
        col_coeff_mat_BC_adj = np.array([0, 1, 0], dtype=c_int)
        id_coeff_mat_BC_adjW = np.array([112, 201, 301], dtype=c_int)
        id_coeff_mat_BC_adjP = np.array([101, 201, 301], dtype=c_int)
        coeff_mat_BC_adjW_idx = fd.analysis.create_matrix(id_coeff_mat_BC_adjW, row_coeff_mat_BC_adj, col_coeff_mat_BC_adj, scale_coeff_mat_BC_adj)
        coeff_mat_BC_adjP_idx = fd.analysis.create_matrix(id_coeff_mat_BC_adjP, row_coeff_mat_BC_adj, col_coeff_mat_BC_adj, scale_coeff_mat_BC_adj)
        self.slot_coeff_mat_BC_adjW = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjW_idx, True)
        self.slot_coeff_mat_BC_adjP = fd.ModelRelatedMatrixBuffer(coeff_mat_BC_adjP_idx, True)
        self.coeff_mat_BC_adjW = getattr(self.slot_coeff_mat_BC_adjW, 'sp_mat') 
        self.coeff_mat_BC_adjP = getattr(self.slot_coeff_mat_BC_adjP, 'sp_mat') 
        
        
        self.init_BDF_sparse_coeff_mat_slot()
# ----------------------------------------------------------------------------- 

    def init_BDF_sparse_coeff_mat_slot(self):
              
        
        
        id_mat = np.array([
            # Row 1: eta0*I | -G | -(Cq v)_q^Tr | Cq^Tr
            112, 128, 130, 125, 122, 202, 201,
            # Row 2: -I | eta0*M - fv^Tr | -Cq^Tr | 0
            112, 101, 129, 201,
            # Row 3: 0 | Cq | 0 | 0
            301,
            # Row 4: Cq | 0 | 0 | 0
            301], dtype=c_int)
        row_pos_mat = np.array([
            # Row 1:
            0, 0, 0, 0, 0, 0, 0,
            # Row 2:
            1, 1, 1, 1,
            # Row 3:
            2,
            # Row 4:
            3], dtype=c_int)
        col_pos_mat = np.array([
            # Row 1:
            0, 1, 1, 1, 1, 2, 3,
            # Row 2:
            0, 1, 1, 2,
            # Row 3:
            1,
            # Row 4:
            0], dtype=c_int)
        BDF_coeff_mat_scale_neu = np.array([
            # Row 1:
            1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
            # Row 2:
            -1.0, 1.0, -1.0, -1.0,
            # Row 3:
            1.0,
            # Row 4:
            1.0])
        mat_idx = fd.analysis.create_matrix(id_mat, row_pos_mat, col_pos_mat, BDF_coeff_mat_scale_neu)
        self.slot_BDF_coeff_mat_neu = fd.ModelRelatedMatrixBuffer(mat_idx, True)
        self.BDF_coeff_mat_neu = getattr(self.slot_BDF_coeff_mat_neu, 'sp_mat')
        self.slot_BDF_coeff_mat_neu.init_scaling(BDF_coeff_mat_scale_neu)

# -----------------------------------------------------------------------------

    def update_sys_mat_for_BDF(self, API):
        API.update_jacobian()
        self.slot_M.update_from_dll()
        # self.slot_M_fv.update_from_dll()
        # self.slot_Cq.update_from_dll()
        # self.slot_CqvDq.update_from_dll()
        # self.slot_G_tr.update_from_dll() 

        self.slot_M.apply_to_cached_matrix()

        self.slot_BDF_coeff_mat_neu.update_from_dll()
        # self.slot_BDF_coeff_mat_neu.apply_to_cached_matrix()
# ----------------------------------------------------------------------------- 