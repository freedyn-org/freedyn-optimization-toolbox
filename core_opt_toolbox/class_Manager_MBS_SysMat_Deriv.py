import freedyn as fd
import numpy as np
from ctypes import c_int

class MBS_SysMat:
    
    def __init__(self,
                 nameParFdu):

        """ Decision on dense or sparse dense_matrices """
        if self.nDof < (10 * 7 + 1):
            self.MBS_modeMAT_sparse = False  
        else:
            self.MBS_modeMAT_sparse = True

        self.create_ID_MBS()    
        self.init_MBS_SysMat_slots(self.MBS_modeMAT_sparse)    
        self.create_views_MBS_SysMat_slots()
        
        """ Derivative of sum of external forces w.r.t. parameter given as string """
        self.nameParFdu = nameParFdu
        self.buffer_MBS_fDu = fd.ForceParameterDerivativeMatrixBuffer(nameParFdu,False)
        self.fDu = self.buffer_MBS_fDu.data
                
        print('class MBS_Sys_mat initialized')

# -----------------------------------------------------------------------------        
        
    def __del__(self):
        
        return None

# -----------------------------------------------------------------------------

    def init_MBS_SysMat_slots(self, modeMAT_sparse):
        
        self.slot_MBS_M = fd.ModelRelatedMatrixBuffer(self.MBS_M_idx, modeMAT_sparse)
        self.slot_MBS_Cq = fd.ModelRelatedMatrixBuffer(self.MBS_Cq_idx, modeMAT_sparse)
        self.slot_MBS_CqvDq = fd.ModelRelatedMatrixBuffer(self.MBS_CqvDq_idx, modeMAT_sparse)
        self.slot_MBS_fv = fd.ModelRelatedMatrixBuffer(self.MBS_fv_idx, modeMAT_sparse)
        self.slot_MBS_G_tr = fd.ModelRelatedMatrixBuffer(self.MBS_G_idx, modeMAT_sparse)

        return None
        
# -----------------------------------------------------------------------------

    def create_views_MBS_SysMat_slots(self):
        
        attr_name = 'sp_mat' if self.MBS_modeMAT_sparse else 'dense_mat'
        
        self.MBS_M = getattr(self.slot_MBS_M, attr_name) 
        self.MBS_Cq = getattr(self.slot_MBS_Cq, attr_name) 
        self.MBS_CqvDq = getattr(self.slot_MBS_CqvDq, attr_name) 
        self.MBS_fv = getattr(self.slot_MBS_fv, attr_name) 
        self.MBS_G_tr = getattr(self.slot_MBS_G_tr, attr_name) 

        return None
        
# -----------------------------------------------------------------------------

    def update_MBS_SysMat_dll_nonzeros(self):
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_G_tr.update_from_dll()
        
        return None         

# -----------------------------------------------------------------------------

    def update_MBS_SysMat(self):
        
        self.update_MBS_SysMat_dll_nonzeros()
        
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        self.slot_MBS_fv.apply_to_cached_matrix()
        self.slot_MBS_G_tr.apply_to_cached_matrix()
        
        return None                 

# -----------------------------------------------------------------------------
        
    def create_ID_MBS(self):
        
        # Mass matrix M
        self.MBS_M_idx = fd.analysis.create_matrix(np.array([101], dtype=c_int), 
                                              np.array([0], dtype=c_int), 
                                              np.array([0], dtype=c_int), 
                                              np.array([1.0])) 
        
        # Constraint Jacobian Cq
        self.MBS_Cq_idx = fd.analysis.create_matrix(np.array([301], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([1.0]))
        
        # CQDT
        self.MBS_CqvDq_idx = fd.analysis.create_matrix(np.array([302], dtype=c_int), 
                                                  np.array([0], dtype=c_int), 
                                                  np.array([0], dtype=c_int), 
                                                  np.array([1.0]))
        
        # fv
        self.MBS_fv_idx = fd.analysis.create_matrix(np.array([109], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([0], dtype=c_int), 
                                               np.array([1.0]))     
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        self.MBS_G_idx = fd.analysis.create_matrix(np.array([108, 110, 105, 102], dtype=c_int), 
                                                   np.array([0, 0, 0, 0], dtype=c_int), 
                                                   np.array([0, 0, 0, 0], dtype=c_int), 
                                                   np.array([1.0, -1.0, -1.0, -1.0]))
        
        return None
    
# -----------------------------------------------------------------------------