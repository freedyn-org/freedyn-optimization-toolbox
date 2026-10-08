from core.BDF_dense_physical_time import DenseBDF
from core.BDF_sparse_physical_time import SparseBDF

from core.consistent_boundary_conditions import DenseBC
from core.consistent_boundary_conditions import SparseBC


def set_up_BDF_layout_sys_mat(dataOpt, FreeDyn):
    
    if FreeDyn.MBS_modeMAT_sparse:
        return SparseBDF(dataOpt, FreeDyn)
    else:
        return DenseBDF(dataOpt, FreeDyn)
    
    
def set_up_consistent_BC_FDOP(dataOpt, FreeDyn):
    
    if FreeDyn.MBS_modeMAT_sparse:
        return SparseBC(dataOpt, FreeDyn)
    else:
        return DenseBC(dataOpt, FreeDyn)