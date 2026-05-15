import freedyn as fd
        

class MBS_SysMat:
    
    def __init__(self):
        
        # Derivative of sum of external forces w.r.t. parameter given as string
        self.buffer_MBS_fDu = fd.ForceParameterDerivativeMatrixBuffer(self.nameParFdu, 
                                                                      False)
        self.fDu = self.buffer_MBS_fDu.data
        
        # self.slot_MBS_states = fd.MBS_States_slots(self.nDof, self.nConstr)    

        """ Decision on dense or sparse dense_matrices """
        if self.nDofConstr < (10 * 7 + 1):
            self.BDF_modeMAT_sparse = False
            
        else:
            self.BDF_modeMAT_sparse = True

            
        self.init_MBS_SysMat_slots(self.BDF_modeMAT_sparse)    
        self.create_views_MBS_SysMat_slots()
        
                
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