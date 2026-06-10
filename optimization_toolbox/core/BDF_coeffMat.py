import numpy as np
import scipy
from scipy.sparse import bmat

# =============================================================================
# BDF System - Matrix layout dense
# =============================================================================

def init_coeffMat_AdjSys_dense(self):
    
    """ Allocate the coefficient matrix and create views of the individual blocks """
    
    nBDFsys = self.nDof + 2*self.nConstr
    self.BDF_coeffMat = np.zeros((nBDFsys, nBDFsys))
    
    block1 = slice(None, self.nDof)
    block2 = slice(self.nDof, self.nDofConstr)
    block3 = slice(self.nDofConstr, nBDFsys)

    self.BDF_coeffMat_view_11 = self.BDF_coeffMat[block1, block1] 
    self.BDF_coeffMat_view_12 = self.BDF_coeffMat[block1, block2]
    self.BDF_coeffMat_view_13 = self.BDF_coeffMat[block1, block3]
    self.BDF_coeffMat_view_21 = self.BDF_coeffMat[block2, block1]
    self.BDF_coeffMat_view_22 = self.BDF_coeffMat[block2, block2]
    self.BDF_coeffMat_view_23 = self.BDF_coeffMat[block2, block3]
    self.BDF_coeffMat_view_31 = self.BDF_coeffMat[block3, block1]
    
    return nBDFsys
# -----------------------------------------------------------------------------

def update_coeffMat_AdjSys_dense(self, eta0, eta0_inv):
    
    """ Update the entries of the coefficient matrix - use the dense matricies """
    
    self.BDF_coeffMat_view_11[:] = eta0 * self.MBS_M - eta0_inv * self.MBS_G_tr.T - self.MBS_fv.T
    self.BDF_coeffMat_view_12[:] = -eta0_inv * self.MBS_CqvDq.T - self.MBS_Cq.T
    self.BDF_coeffMat_view_13[:] = -eta0_inv * self.MBS_Cq.T
    self.BDF_coeffMat_view_21[:] = self.MBS_Cq @ self.MBS_G_tr.T
    self.BDF_coeffMat_view_22[:] = self.MBS_Cq @ self.MBS_CqvDq.T
    self.BDF_coeffMat_view_23[:] = self.MBS_Cq @ self.MBS_Cq.T
    self.BDF_coeffMat_view_31[:] = self.MBS_Cq
# -----------------------------------------------------------------------------

# =============================================================================
# BDF System - Matrix layout sparse
# =============================================================================

def init_coeffMat_AdjSys_sparse(self, formatMAT):
    
    """ Allocate the coefficient matrix and create maps of the individual blocks """
    
    nBDFsys = 2 * (self.nDof + self.nConstr)

    self.update_MBS_SysMat() 
    eyeMat_sp = scipy.sparse.eye(self.nDof, format='csr')
    
    dummy_M = self.slot_MBS_M.sp_mat.copy()
    dummy_fv = self.slot_MBS_fv.sp_mat.copy()
    dummy_M.data.fill(1.0)
    dummy_fv.data.fill(1.0)        
    sumA22 = dummy_M + dummy_fv.T

    offset = 0
    offset, A11_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, eyeMat_sp)  
    offset, A12_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_G_tr, transpose=True) 
    offset, A13_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_CqvDq, transpose=True)
    offset, A14_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq, transpose=True) 
    offset, A21_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, eyeMat_sp)
    offset, A22_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, sumA22)        
    offset, A23_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq, transpose=True)
    offset, A32_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq)  
    offset, A41_idx = idx_temp_init_coeffMat_AdjSys_sparse(offset, self.MBS_Cq)
    
    layout = [[A11_idx, A12_idx, A13_idx, A14_idx],
              [A21_idx, A22_idx, A23_idx, None],
              [None, A32_idx, None, None],
              [A41_idx, None, None, None]]
    
    # Bulid Coeff Mat as csc sparse matrix
    self.BDF_spCoeffMat = bmat(layout, format=formatMAT).astype(np.float64)
    
    # Mapping
    self.BDF_spCoeffMat_map = np.argsort(self.BDF_spCoeffMat.data)

    self.BDF_coeffMat_map_A11 = self.BDF_spCoeffMat_map[build_map_coeffMAT(eyeMat_sp, A11_idx)]
    self.BDF_coeffMat_map_A12 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_G_tr, A12_idx, transpose=True)]
    self.BDF_coeffMat_map_A13 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_CqvDq, A13_idx, transpose=True)]
    self.BDF_coeffMat_map_A14 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_Cq, A14_idx, transpose=True)]
    self.BDF_coeffMat_map_A21 = self.BDF_spCoeffMat_map[build_map_coeffMAT(eyeMat_sp, A21_idx)]
    self.BDF_coeffMat_map_M_A22 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_M, A22_idx)]
    self.BDF_coeffMat_map_fv_A22 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_fv, A22_idx, transpose=True)]
    self.BDF_coeffMat_map_A22 = np.unique(np.concatenate([self.BDF_coeffMat_map_M_A22,self.BDF_coeffMat_map_fv_A22]))
    self.BDF_coeffMat_map_A23 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_Cq, A23_idx, transpose=True)]
    self.BDF_coeffMat_map_A32 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_Cq, A32_idx)]
    self.BDF_coeffMat_map_A41 = self.BDF_spCoeffMat_map[build_map_coeffMAT(self.MBS_Cq, A41_idx)]
    
    # A21 is set here, as these are const. values
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A21] = -1.0
    
    return nBDFsys
# -----------------------------------------------------------------------------

def idx_temp_init_coeffMat_AdjSys_sparse(offset, mtx, transpose=False):
    
    tpl = mtx.T.copy() if transpose else mtx.copy()
    tpl.data = np.arange(offset, offset + tpl.nnz, dtype=int) + 1
    offset += tpl.nnz       
    return offset, tpl
# -----------------------------------------------------------------------------        
    
def build_map_coeffMAT(sub_mat, block_idx_mat, transpose=False):
    
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

def update_coeffMat_AdjSys_sparse(self, eta0):
    
    """ Update the entries of the coefficient matrix - use directly the non-zeros elements from the API """
    
    # Row 1
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A11] = eta0 
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A12] = -self.slot_MBS_G_tr.dll_nonzeros
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A13] = -self.slot_MBS_CqvDq.dll_nonzeros
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A14] = -self.slot_MBS_Cq.dll_nonzeros
    
    # Row 2
    # A21 is already set in self.init_coeffMat_AdjSys_sparse
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A22] = 0.0
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_M_A22] = eta0 * self.slot_MBS_M.dll_nonzeros
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_fv_A22] -= self.slot_MBS_fv.dll_nonzeros
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A23] = -self.slot_MBS_Cq.dll_nonzeros

    # Row 3
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A32] = self.slot_MBS_Cq.dll_nonzeros 
    
    # Row 4
    self.BDF_spCoeffMat.data[self.BDF_coeffMat_map_A41] = self.slot_MBS_Cq.dll_nonzeros 
# -----------------------------------------------------------------------------