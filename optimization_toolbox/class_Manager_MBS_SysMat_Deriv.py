import freedyn as fd
import numpy as np
from ctypes import c_int

class MBS_SysMat:
    
    def __init__(self,
                 nameParFdu):

        """ System matrices and derivatives - Decision: dense or sparse layout """
        if self.nDof < (10 * 7 + 1):
            self.MBS_modeMAT_sparse = False  
        else:
            self.MBS_modeMAT_sparse = True
            
        """ System matrices and derivatives - Decision: dense or sparse layout """
        self.init_MBS_SysMat_slots()
        
        """ Derivative of sum of external forces w.r.t. parameter given as string """
        self.buffer_MBS_fDu = fd.ForceParameterDerivativeMatrixBuffer(nameParFdu)
        self.fDu = self.buffer_MBS_fDu.data
                
        print('class MBS_Sys_mat initialized')

# -----------------------------------------------------------------------------        
        
    def __del__(self):
        return None

# -----------------------------------------------------------------------------

    def init_MBS_SysMat_slots(self):
        
        attr_name = 'sp_mat' if self.MBS_modeMAT_sparse else 'dense_mat'
        
        # Mass matrix M
        M_idx = fd.analysis.create_matrix(np.array([101], dtype=c_int), 
                                              np.array([0], dtype=c_int), 
                                              np.array([0], dtype=c_int), 
                                              np.array([1.0]))
        self.slot_MBS_M = fd.ModelRelatedMatrixBuffer(M_idx, self.MBS_modeMAT_sparse)
        self.MBS_M = getattr(self.slot_MBS_M, attr_name) 
        
        # Constraint Jacobian Cq
        Cq_idx = fd.analysis.create_matrix(np.array([301], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([1.0]))
        self.slot_MBS_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, self.MBS_modeMAT_sparse)
        self.MBS_Cq = getattr(self.slot_MBS_Cq, attr_name) 
        
        # CQDT
        CqvDq_idx = fd.analysis.create_matrix(np.array([302], dtype=c_int), 
                                                  np.array([0], dtype=c_int), 
                                                  np.array([0], dtype=c_int), 
                                                  np.array([1.0]))
        self.slot_MBS_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, self.MBS_modeMAT_sparse)
        self.MBS_CqvDq = getattr(self.slot_MBS_CqvDq, attr_name) 
        
        # fv
        fv_idx = fd.analysis.create_matrix(np.array([109], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([1.0]))
        self.slot_MBS_fv = fd.ModelRelatedMatrixBuffer(fv_idx, self.MBS_modeMAT_sparse)
        self.MBS_fv = getattr(self.slot_MBS_fv, attr_name) 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        G_idx = fd.analysis.create_matrix(np.array([108, 110, 105, 102], dtype=c_int), 
                                                   np.array([0, 0, 0, 0], dtype=c_int), 
                                                   np.array([0, 0, 0, 0], dtype=c_int), 
                                                   np.array([1.0, -1.0, -1.0, -1.0]))
        self.slot_MBS_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, self.MBS_modeMAT_sparse)
        self.MBS_G_tr = getattr(self.slot_MBS_G_tr, attr_name) 
        
# -----------------------------------------------------------------------------

    def update_MBS_SysMat_dll_nonzeros(self):
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_G_tr.update_from_dll()      

# -----------------------------------------------------------------------------

    def update_MBS_SysMat(self):
        
        self.update_MBS_SysMat_dll_nonzeros()
        
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        self.slot_MBS_fv.apply_to_cached_matrix()
        self.slot_MBS_G_tr.apply_to_cached_matrix()               

# -----------------------------------------------------------------------------