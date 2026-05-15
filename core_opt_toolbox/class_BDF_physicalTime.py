import numpy as np
import freedyn as fd

import scipy
from scipy.sparse.linalg import factorized
from scipy.sparse import bmat


from class_BDF_intOrderOne_physicalTime import BDF_intOrderOne 
from class_BDF_intOrderTwo_physicalTime import BDF_intOrderTwo 

from AdjSys_coeffMat_dense import CoeffMat
from AdjSys_coeffMat_sparse import spCoeffMat

class BDF(BDF_intOrderOne, BDF_intOrderTwo, CoeffMat, spCoeffMat):
    
    def __init__(self):
        
        self.BDF_idx_buff = 0
        self.BDF_diff_tau = np.zeros(2)
        self.adjW_J_buff = np.empty((2, self.nDof))
        self.adjP_J_buff = np.empty((2, self.nDof))
        self.BDF_J_Mp_buff = np.zeros((2, self.nDof))
        
        self.adjW_Phi_buff = np.empty((2, self.nDof, self.num_xF))
        self.adjP_Phi_buff = np.empty((2, self.nDof, self.num_xF))
        self.BDF_Phi_Mp_buff = np.zeros((2, self.nDof, self.num_xF))
               
        
        self.BDF_BC_dq_tr = np.zeros((self.nDofConstr, self.num_xF))   # (dPhi / dq)^T
        self.BDF_BC_dv_tr = np.zeros((self.nDofConstr, self.num_xF))   # (dPhi / dv)^T        
        
            
        if self.BDF_modeMAT_sparse:   
            nBDFsys = spCoeffMat.__init__(self)
            self.BDF_BC_eyeMat = scipy.sparse.eye(self.nDof)
            self.compute_consistent_BC_J = self.compute_consistent_BC_J_sparse
            self.compute_consistent_BC_Phi = self.compute_consistent_BC_Phi_sparse

        else:
            nBDFsys = CoeffMat.__init__(self)
            self.compute_consistent_BC_J = self.compute_consistent_BC_J_dense
            self.compute_consistent_BC_Phi = self.compute_consistent_BC_Phi_dense
            self.BDF_BC_eyeMat = np.eye(self.nDof)
            self.BDF_BC_zeroMat = np.zeros((self.nConstr, self.nConstr))

        
        self.BDF_solVec_J = np.zeros(nBDFsys)
        self.BDF_solVec_Phi = np.zeros((nBDFsys, self.num_xF))

        BDF_intOrderOne.__init__(self)
        BDF_intOrderTwo.__init__(self)
           
        print('class BDF initialized')      
        
# -----------------------------------------------------------------------------
    
    def get_consistent_BC_J(self):
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        
        self.compute_consistent_BC_J()
        
        self.adjP_J_buff[self.BDF_idx_buff, :].fill(0.0)
        self.adjW_J_buff[self.BDF_idx_buff, :].fill(0.0)
        
        return None
        
# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_dense(self):

        return None 

# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_J_sparse(self):

        return None

# -----------------------------------------------------------------------------
    
    def get_consistent_BC_Phi(self):
        
        self.get_Phi_dq()    # (dPhi / dq)^T
        self.get_Phi_dv()    # (dPhi / dv)^T
        
        self.BDF_BC_dq_tr[:self.nDof,:] = self.dPhidq.T    # (dPhi / dq)^T
        self.BDF_BC_dv_tr[:self.nDof,:] = self.dPhidv.T    # (dPhi / dv)^T        
        
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        
        self.BDF_idx_buff = 0
        
        
        WL_tF, PU_tF = self.compute_consistent_BC_Phi()
        
        self.adjW_Phi_buff[self.BDF_idx_buff, :, :] = WL_tF[:self.nDof, :]     # W_tF
        self.adjP_Phi_buff[self.BDF_idx_buff, :, :] = PU_tF[:self.nDof, :]     # P_tF

        return None

# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_dense(self):        

        coeffMat_W = np.block([[self.BDF_BC_eyeMat, self.MBS_Cq.T],
                                   [self.MBS_Cq, self.BDF_BC_zeroMat]])
        
        
        coeffMat_P = np.block([[self.MBS_M, self.MBS_Cq.T],
                               [self.MBS_Cq, self.BDF_BC_zeroMat]])

        
        PU_tF = np.linalg.solve(coeffMat_P, self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.nDof:, :]
        
        
        self.BDF_BC_dq_tr[:self.nDof, :] -= (self.MBS_CqvDq.T @ U_tF)
         
        WL_tF = np.linalg.solve(coeffMat_W, self.BDF_BC_dq_tr)

        return WL_tF, PU_tF

# -----------------------------------------------------------------------------
    
    def compute_consistent_BC_Phi_sparse(self):        
        
        coeffMat_W_csc = bmat([[self.BDF_BC_eyeMat, self.MBS_Cq.T],
                               [self.MBS_Cq, None]], format = 'csc')
        
        
        coeffMat_P_csc = bmat([[self.M0_csr, self.MBS_Cq.T],
                               [self.MBS_Cq, None]], format = 'csc')
        
        
        solve_W = factorized(coeffMat_W_csc)
        solve_P = factorized(coeffMat_P_csc)


        PU_tF = solve_P(self.BDF_BC_dv_tr)
        U_tF = PU_tF[self.nDof:, :]
        
        
        self.BDF_BC_dq_tr[:self.nDof, :] -= (self.MBS_CqvDq.T @ U_tF)
         
        WL_tF = solve_W(self.BDF_BC_dq_tr)

        return WL_tF, PU_tF

# -----------------------------------------------------------------------------

    def update_userFcts_BDF(self, z, time):
        
        self.get_LagrangianOCP_dq(z,time)
        self.get_LagrangianOCP_dv(z,time)
        
        return None

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
    
    def apply_BDF_J(self, points, finalTime, z):
                
        p = np.empty((self.nDof, points))
        
        self.get_consistent_BC_J() 
        p[:,-1] = self.adjP_J_buff[self.BDF_idx_buff, :]

        """ BDF order 1 """
        tLeft = self.fdApi2.getStatesAtTimeIndex(points-2, self.FDstates) #attention call fdApi2! not available in fdApi
        self.fdApi.updateSystem(tLeft, self.FDstates)
        self.BDF_diff_tau[self.BDF_idx_buff] = finalTime - tLeft
        self.integrate_J_BDF1(z, tLeft)
        p[:,points-2] = self.adjP_J_buff[self.BDF_idx_buff, :]
        
        
        """ BDF order 2 """
        for n in range(points-3, -1, -1):
            tRight = tLeft
            tLeft = self.fdApi2.getStatesAtTimeIndex(n, self.FDstates) #attention call fdApi2! not available in fdApi
            self.fdApi.updateSystem(tLeft, self.FDstates)    
            self.BDF_diff_tau[self.BDF_idx_buff] = tRight - tLeft                 
            self.integrate_J_BDF2(z, tLeft)
            p[:,n] = self.adjP_J_buff[self.BDF_idx_buff, :]

        return p

# -----------------------------------------------------------------------------
    
    def apply_BDF_Phi(self, repeats, points, finalTime):
        
        p = np.empty((self.nDof,repeats,points))
        
        self.get_consistent_BC_Phi() 
        p[:,:,-1] = self.adjP_Phi_buff[self.BDF_idx_buff, :, :]  
         
               
        """ BDF order 1 """
        tLeft = self.fdApi2.getStatesAtTimeIndex(points-2, self.FDstates) #attention call fdApi2! not available in fdApi
        self.fdApi.updateSystem(tLeft, self.FDstates)
        self.BDF_diff_tau[self.BDF_idx_buff] = finalTime - tLeft
        self.integrate_Phi_BDF1(tLeft)
        p[:,:,points-2] = self.adjP_Phi_buff[self.BDF_idx_buff, :, :]  
        
            
        """ BDF order 2 """
        for n in range(points-3, -1, -1):
            tRight = tLeft
            tLeft = self.fdApi2.getStatesAtTimeIndex(n, self.FDstates) #attention call fdApi2! not available in fdApi
            self.fdApi.updateSystem(tLeft, self.FDstates)
            self.BDF_diff_tau[self.BDF_idx_buff] = tRight - tLeft
            self.integrate_Phi_BDF2(tLeft)
            p[:,:,n] = self.adjP_Phi_buff[self.BDF_idx_buff, :, :]  

        return p

# -----------------------------------------------------------------------------

    def singleStep_BDForder_one_J(self, z, time):

        return self.integrate_J_BDF1(z, time)   

# -----------------------------------------------------------------------------

    def singleStep_BDForder_two_J(self, z, time):

        return self.integrate_J_BDF2(z, time)

# -----------------------------------------------------------------------------
    
    def singleStep_BDForder_one_Phi(self, time):

        return self.integrate_Phi_BDF1(time)
    
# -----------------------------------------------------------------------------
    
    def singleStep_BDForder_two_Phi(self, time):

        return self.integrate_Phi_BDF2(time)
    
# -----------------------------------------------------------------------------
