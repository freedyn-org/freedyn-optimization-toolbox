import freedyn as fd
import numpy as np
from ctypes import c_int

class MBS_SysMat:
    
    def __init__(self,
                 nameParFdu):
        
        self.create_identy_matMBS()
        self.create_ID_MBS()
        
        """ Decision on dense or sparse dense_matrices """
        if self.nDofConstr < (10 * 7 + 1):
            self.BDF_modeMAT_sparse = False
            
        else:
            self.BDF_modeMAT_sparse = True

            
        self.init_MBS_SysMat_slots(self.BDF_modeMAT_sparse)    
        self.create_views_MBS_SysMat_slots()
        
        
        """ Derivative of sum of external forces w.r.t. parameter given as string """
        self.nameParFdu = nameParFdu
        self.buffer_MBS_fDu = fd.ForceParameterDerivativeMatrixBuffer(nameParFdu, False)
        self.fDu = self.buffer_MBS_fDu.data
        
                
        print('class MBS_sysdense_mat initialized')
        
        
    def __del__(self):
        
        return None

# -----------------------------------------------------------------------------

    def init_MBS_SysMat_slots(self, modeMAT_sparse):
        
        self.slot_MBS_M = fd.ModelRelatedMatrixBuffer(self.ID_M, modeMAT_sparse)
        self.slot_MBS_Cq = fd.ModelRelatedMatrixBuffer(self.ID_Cq, modeMAT_sparse)
        self.slot_MBS_CqvDq = fd.ModelRelatedMatrixBuffer(self.ID_CqvDq, modeMAT_sparse)
        self.slot_MBS_fv = fd.ModelRelatedMatrixBuffer(self.ID_fv, modeMAT_sparse)
        self.slot_MBS_G_tr = fd.ModelRelatedMatrixBuffer(self.ID_G, modeMAT_sparse)

        return None
        
# -----------------------------------------------------------------------------

    def create_views_MBS_SysMat_slots(self):
        
        attr_name = 'sp_mat' if self.BDF_modeMAT_sparse else 'dense_mat'
        
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
        
    def update_MBS_SysMat_idx(self):

        self.slot_MBS_M.set_index(self.ID_M)
        self.slot_MBS_Cq.set_index(self.ID_Cq)        
        self.slot_MBS_CqvDq.set_index(self.ID_CqvDq)
        self.slot_MBS_fv.set_index(self.ID_fv)
        self.slot_MBS_G_tr.set_index(self.ID_G)             
        
        return None

# -----------------------------------------------------------------------------

    def create_identy_matMBS(self):
        
        self.MBS_singleMat_row = np.array([0], dtype=c_int)
        self.MBS_singleMat_col = np.array([0], dtype=c_int)
        self.MBS_singleMat_scale = np.array([1.0])
        
        self.identy_MBS_MASS = np.array([101], dtype=c_int)     # "MASS"
        self.identy_MBS_CQ = np.array([301], dtype=c_int)          # "CQ"
        self.identy_MBS_CQDT = np.array([302], dtype=c_int)        # "CQDT"
        self.identy_MBS_DQEXTDQD = np.array([109], dtype=c_int)    # "DQEXTDQD"
        
        # G^T = "DQEXTDQ" - "DCQTLEDQ" - "DCQTLIDQ" - "DMQDDDQ"
        self.MBS_G_tr_row = np.array([0, 0, 0, 0], dtype=c_int)
        self.MBS_G_tr_col = np.array([0, 0, 0, 0], dtype=c_int)
        self.MBS_G_tr_scale = np.array([1.0, -1.0, -1.0, -1.0])
        self.identy_MBS_G_tr = np.array([108, 110, 105, 102], dtype=c_int)

        return None


# -----------------------------------------------------------------------------
        
    def create_ID_MBS(self):

        # Mass matrix M
        self.ID_M = fd.analysis.create_matrix(self.identy_MBS_MASS, 
                                              self.MBS_singleMat_row, 
                                              self.MBS_singleMat_col, 
                                              self.MBS_singleMat_scale) 
        
        # Constraint Jacobian Cq
        self.ID_Cq = fd.analysis.create_matrix(self.identy_MBS_CQ, 
                                               self.MBS_singleMat_row, 
                                               self.MBS_singleMat_col, 
                                               self.MBS_singleMat_scale)
        
        # MBS_CQDT
        self.ID_CqvDq = fd.analysis.create_matrix(self.identy_MBS_CQDT, 
                                                  self.MBS_singleMat_row, 
                                                  self.MBS_singleMat_col, 
                                                  self.MBS_singleMat_scale)
        
        # fv
        self.ID_fv = fd.analysis.create_matrix(self.identy_MBS_DQEXTDQD, 
                                               self.MBS_singleMat_row, 
                                               self.MBS_singleMat_col, 
                                               self.MBS_singleMat_scale)     
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        self.ID_G = fd.analysis.create_matrix(self.identy_MBS_G_tr, 
                                              self.MBS_G_tr_row, 
                                              self.MBS_G_tr_col, 
                                              self.MBS_G_tr_scale)
        
        return None
    
# -----------------------------------------------------------------------------