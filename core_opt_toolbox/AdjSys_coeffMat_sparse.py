import numpy as np

import scipy
from scipy.sparse import bmat

class spCoeffMat():
    
    def __init__(self):
        
        nBDFsys = 2*self.nDof+2*self.nConstr
        self.init_coeffMat_AdjSys_sparse('csc')
        
        return nBDFsys
        
        
# -----------------------------------------------------------------------------

    def init_coeffMat_AdjSys_sparse(self, formatMAT):

        self.update_MBS_SysMat() 
        eyeMat_sp = scipy.sparse.eye(self.nDof, format='csr')
        
        dummy_M = self.slot_MBS_M.sp_mat.copy()
        dummy_fv = self.slot_MBS_fv.sp_mat.copy()
        dummy_M.data.fill(1.0)
        dummy_fv.data.fill(1.0)        
        sumA22 = dummy_M + dummy_fv.T

        
        offset = 0
        
        offset, A11_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, eyeMat_sp)  
        offset, A12_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_G_tr, transpose=True) 
        offset, A13_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_CqvDq, transpose=True)
        offset, A14_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq, transpose=True) 
        offset, A21_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, eyeMat_sp)
        offset, A22_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, sumA22)        
        offset, A23_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq, transpose=True)
        offset, A32_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq)  
        offset, A41_idx = self.idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq)
        
        layout = [[A11_idx, A12_idx, A13_idx, A14_idx],
                  [A21_idx, A22_idx, A23_idx, None],
                  [None, A32_idx, None, None],
                  [A41_idx, None, None, None]]
        
        # Bulid Coeff Mat as csc sparse matrix
        self.BDF_spCoeffMat = bmat(layout, format=formatMAT).astype(np.float64)
        
        # Mapping
        self.BDF_spCoeffMat_map = np.argsort(self.BDF_spCoeffMat.data)
        

        self.map_A11 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(eyeMat_sp, A11_idx)]
        self.map_A12 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_G_tr, A12_idx, transpose=True)]
        self.map_A13 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_CqvDq, A13_idx, transpose=True)]
        self.map_A14 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_Cq, A14_idx, transpose=True)]
        self.map_A21 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(eyeMat_sp, A21_idx)]
        self.map_M_A22 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_M, A22_idx)]
        self.map_fv_A22 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_fv, A22_idx, transpose=True)]
        self.map_A22 = np.unique(np.concatenate([self.map_M_A22,self.map_fv_A22]))
        self.map_A23 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_Cq, A23_idx, transpose=True)]
        self.map_A32 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_Cq, A32_idx)]
        self.map_A41 = self.BDF_spCoeffMat_map[self.build_map_coeffMAT(self.MBS_Cq, A41_idx)]
        
        
        self.BDF_spCoeffMat.data[self.map_A21] = -1.0

        return None

# -----------------------------------------------------------------------------

    def idx_temp_init_coeffMat_AdjSys_sparse(self, offset, mtx, transpose=False):
        
        tpl = mtx.T.copy() if transpose else mtx.copy()
        tpl.data = np.arange(offset, offset + tpl.nnz, dtype=int) + 1
        offset += tpl.nnz       
        return offset, tpl

# -----------------------------------------------------------------------------        
    
    def build_map_coeffMAT(self, sub_mat, block_idx_mat, transpose=False):
        
        num_cols = block_idx_mat.shape[1]
        
        
        target_coo = block_idx_mat.tocoo()
        target_keys = target_coo.row * num_cols + target_coo.col
        sort_idx = np.argsort(target_keys)
        sorted_target_keys = target_keys[sort_idx]
        
        
        sub_coo = sub_mat.tocoo()
        
        if transpose:
            sub_keys = sub_coo.col * num_cols + sub_coo.row
            
        else:
            sub_keys = sub_coo.row * num_cols + sub_coo.col
        
        
        matched_pos = np.searchsorted(sorted_target_keys, sub_keys)
        
        
        return (block_idx_mat.data[sort_idx[matched_pos]] - 1).astype(np.int32)
        
# -----------------------------------------------------------------------------

    def get_coeffMat_AdjSys_sparse(self, eta0):
        
        # Row 1
        self.BDF_spCoeffMat.data[self.map_A11] = eta0 
        self.BDF_spCoeffMat.data[self.map_A12] = -self.slot_MBS_G_tr.dll_nonzeros
        self.BDF_spCoeffMat.data[self.map_A13] = -self.slot_MBS_CqvDq.dll_nonzeros
        self.BDF_spCoeffMat.data[self.map_A14] = -self.slot_MBS_Cq.dll_nonzeros
        
        # Row 2
        # A21 is already set in self.init_coeffMat_AdjSys_sparse
        self.BDF_spCoeffMat.data[self.map_A22] = 0.0
        self.BDF_spCoeffMat.data[self.map_M_A22] = eta0 * self.slot_MBS_M.dll_nonzeros
        self.BDF_spCoeffMat.data[self.map_fv_A22] -= self.slot_MBS_fv.dll_nonzeros
        self.BDF_spCoeffMat.data[self.map_A23] = -self.slot_MBS_Cq.dll_nonzeros

        # Row 3
        self.BDF_spCoeffMat.data[self.map_A32] = self.slot_MBS_Cq.dll_nonzeros 
        
        # Row 4
        self.BDF_spCoeffMat.data[self.map_A41] = self.slot_MBS_Cq.dll_nonzeros 

        return None

# -----------------------------------------------------------------------------