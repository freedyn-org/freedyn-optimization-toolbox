import numpy as np
import scipy
from scipy.sparse.linalg import factorized

from consistent_Boundary_conditions import BC_consistent
from BDF_intOrderOne_physicalTime import BDF_intOrderOne 
from BDF_intOrderTwo_physicalTime import BDF_intOrderTwo 

from AdjSys_coeffMat import CoeffMat

class BDF(BC_consistent, BDF_intOrderOne, BDF_intOrderTwo, CoeffMat):
    
    def __init__(self):
        
        self.BDF_idx_buff = 0
        self.BDF_diff_tau = np.zeros(2)
        
        self.adjW_J_buff = np.empty((2, self.nDof))
        self.adjP_J_buff = np.empty((2, self.nDof))
        self.BDF_J_Mp_buff = np.zeros((2, self.nDof))
        
        self.adjW_Phi_buff = np.empty((2, self.nDof, self.num_xF))
        self.adjP_Phi_buff = np.empty((2, self.nDof, self.num_xF))
        self.BDF_Phi_Mp_buff = np.zeros((2, self.nDof, self.num_xF))
               
        nBDFsys = CoeffMat.__init__(self)
        self.BDF_solVec_J = np.zeros(nBDFsys)
        self.BDF_solVec_Phi = np.zeros((nBDFsys, self.num_xF))
        
        BC_consistent.__init__(self)
        BDF_intOrderOne.__init__(self)
        BDF_intOrderTwo.__init__(self)
           
        print('class BDF initialized')      
        
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

    def update_userFcts_BDF(self, z):
        self.get_LagrangianOCP_dq(z)
        self.get_LagrangianOCP_dv(z)

# -----------------------------------------------------------------------------

    def solve_J_AdjSys_dense(self):
        return np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_J)
    
# -----------------------------------------------------------------------------

    def solve_Phi_AdjSys_dense(self):
        return np.linalg.solve(self.BDF_coeffMat, self.BDF_solVec_Phi)   

# -----------------------------------------------------------------------------

    def solve_J_AdjSys_sparse(self):
        solve = factorized(self.BDF_spCoeffMat)
        return solve(self.BDF_solVec_J)
    
# -----------------------------------------------------------------------------

    def solve_Phi_AdjSys_sparse(self):
        solve = factorized(self.BDF_spCoeffMat)
        return solve(self.BDF_solVec_Phi)

# -----------------------------------------------------------------------------