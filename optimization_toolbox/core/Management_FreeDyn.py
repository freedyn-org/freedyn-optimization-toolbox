import numpy as np
import freedyn as fd
from ctypes import c_int
from pathlib import Path


class FreeDyn():
    
    def __init__(self,
                 path_FDdll, path_fds, name_fds,
                 name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                 num_FD_pars):
        
        # Initialize FreeDyn API
        fd.initialize(path_FDdll)

        # Define path and name of *.fds
        self.fds_path = path_fds
        self.fds_path_name = str(Path(path_fds) / f'{name_fds}.fds')
       
        
        # Load and Read *.fds
        self.load_and_read_fds(self.fds_path_name)
        self.fds_set_writing_to_none(self.fds_path_name) # set any file writing of FreeDyn to no
        
        # Create Model
        self.API = fd.Model(self.fds_path_name, status_output="NO")
        info = self.API.get_info()
        self.nDof = info.num_generalized_coordinates
        self.nConstr = info.num_lagrange_multipliers
        self.nDofConstr = self.nDof + self.nConstr 
        self.num_time_steps = 0
        
        # System matrices and derivatives - Decision: dense or sparse layout
        lim_val = 10 * 7
        if self.nDof > lim_val:
            self.MBS_modeMAT_sparse = True
            print("Matrix layout: sparse")
        else:
            self.MBS_modeMAT_sparse = False  
            print("Matrix layout: dense")
            
            
        # System matrices and derivatives - Decision: dense or sparse layout
        self.init_MBS_sysMat_slots()
        
        # Derivative of sum of external forces w.r.t. parameter given as string
        self.buffer_MBS_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(name_dForce_dparam)
        self.dForce_dFDparam = self.buffer_MBS_dForce_dFDparam.data
        
        # FreeDyn data object control splines, measures, force wrt parameter
        self.name_ctrlSPL = name_ctrlSPL
        self.name_meas = name_fDmeas
        self.name_dForce_dparam = name_dForce_dparam
        self.FD_pars = None
        self.num_FD_pars = num_FD_pars
        
        print('class FreeDyn initialized')
# -----------------------------------------------------------------------------        
        
    def delete_model(self):
        
        self.API.__del__()
        print('Model deleted')

# =============================================================================
# Commands concerning system state
# =============================================================================        
        
    def fetch_and_update_states_at_index(self, idx):
        self.API.fetch_states_at_index(idx)
        self.API.update_state_at_index(idx) # necessary, if measures are used in get_lagrangian()
# -----------------------------------------------------------------------------  
        
# =============================================================================
# Commands concerning system matrices
# =============================================================================

    def init_MBS_sysMat_slots(self):
        
        # Row/Column position and scaling value of single matrix
        pos_mat = np.array([0], dtype=c_int)
        scale_mat = np.array([1.0])
        
        # Set up the memory layout either as sparse or dense
        attr_name = 'sp_mat' if self.MBS_modeMAT_sparse else 'dense_mat'
        
        # Mass matrix M
        id_M = np.array([101], dtype=c_int)
        M_idx = fd.analysis.create_matrix(id_M, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_M = fd.ModelRelatedMatrixBuffer(M_idx, self.MBS_modeMAT_sparse)
        self.MBS_M = getattr(self.slot_MBS_M, attr_name) 
        
        # Constraint Jacobian Cq
        id_Cq = np.array([301], dtype=c_int)
        Cq_idx = fd.analysis.create_matrix(id_Cq, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_Cq = fd.ModelRelatedMatrixBuffer(Cq_idx, self.MBS_modeMAT_sparse)
        self.MBS_Cq = getattr(self.slot_MBS_Cq, attr_name) 
        
        # CQDT
        id_CqvDq = np.array([302], dtype=c_int)
        CqvDq_idx = fd.analysis.create_matrix(id_CqvDq, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_CqvDq = fd.ModelRelatedMatrixBuffer(CqvDq_idx, self.MBS_modeMAT_sparse)
        self.MBS_CqvDq = getattr(self.slot_MBS_CqvDq, attr_name) 
        
        # fv
        id_fv = np.array([109], dtype=c_int)
        fv_idx = fd.analysis.create_matrix(id_fv, pos_mat, pos_mat, scale_mat)
        self.slot_MBS_fv = fd.ModelRelatedMatrixBuffer(fv_idx, self.MBS_modeMAT_sparse)
        self.MBS_fv = getattr(self.slot_MBS_fv, attr_name) 
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        id_matG = np.array([108, 110, 105, 102], dtype=c_int)
        pos_matG = np.array([0, 0, 0, 0], dtype=c_int)
        scale_matG = np.array([1.0, -1.0, -1.0, -1.0])
        G_idx = fd.analysis.create_matrix(id_matG, pos_matG, pos_matG, scale_matG)
        self.slot_MBS_G_tr = fd.ModelRelatedMatrixBuffer(G_idx, self.MBS_modeMAT_sparse)
        self.MBS_G_tr = getattr(self.slot_MBS_G_tr, attr_name)    
# ----------------------------------------------------------------------------- 

    def update_sys_mat_for_BDF(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_fv.apply_to_cached_matrix()
        self.slot_MBS_G_tr.update_from_dll()   
        self.slot_MBS_G_tr.apply_to_cached_matrix()
# ----------------------------------------------------------------------------- 

    def update_nnz_dll_sys_mat_for_BDF(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_fv.update_from_dll()
        self.slot_MBS_G_tr.update_from_dll()   
# -----------------------------------------------------------------------------
    
    def update_sys_mat_for_consistent_BC_J(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
# -----------------------------------------------------------------------------

    def update_sys_mat_for_consistent_BC_Phi(self):
        self.API.update_jacobian()
        self.slot_MBS_M.update_from_dll()
        self.slot_MBS_M.apply_to_cached_matrix()
        self.slot_MBS_Cq.update_from_dll()
        self.slot_MBS_Cq.apply_to_cached_matrix()
        self.slot_MBS_CqvDq.update_from_dll()
        self.slot_MBS_CqvDq.apply_to_cached_matrix()
# -----------------------------------------------------------------------------

# =============================================================================
# Commands concerning FD pars 
# =============================================================================
    
    def update_FD_pars(self, param_names, values):
        
        for name, val in zip(param_names, values): #, strict=True
            self.API.set_parameter(name, val)
 
# =============================================================================
# Commands concerning splines
# =============================================================================
    
    def update_ctrl_spline(self, dataOpt, Ctrl):
        
        realT = dataOpt.final_time * Ctrl.grid_tau
        
        for i, SPL in enumerate(self.name_ctrlSPL):
            self.API.set_spline(SPL, realT, Ctrl.grid_nodes[:,i])
# -----------------------------------------------------------------------------              
            
    def write_ctrl_dataSPL(self, Ctrl):
        
        realT = Ctrl.final_time * Ctrl.grid_tau
        data = np.column_stack((realT, Ctrl.grid_nodes))
        np.savetxt(f'{self.fds_path}\\dataSPL.txt', data, fmt='%.10f')   

# =============================================================================
# Commands concerning file.fds
# =============================================================================
   
    def load_and_read_fds(self, fds):
        
        keys = {"isExactOutputTimeEnforced",
                "WriteConstraintForceResultFile",
                "WriteForceResultFile",
                "WriteVelocityResultFile",
                "WriteStateResultFile",
                "WriteAccelerationResultFile",
                "WriteExtConstraintLagrangeResultFile",
                "WriteMeasureResultFile"}
        self.fds_idxLine = dict.fromkeys(keys, None)
        
        # Open FDS file and store data
        with open(fds, 'r') as inp:
           self.fds_data = inp.readlines()      
        
        # Get idex of lines
        for i, line in enumerate(self.fds_data):
            for key in keys:
                if line.lstrip().startswith(key):
                    self.fds_idxLine[key] = i
# -----------------------------------------------------------------------------    

    def write_fds(self, fds):
        
        with open(fds, 'w') as target:
           target.writelines(self.fds_data)           
# ---------------------------------------------------------------------------- 

    def fds_set_writing_to_none(self, fds):
        
        self.fds_data[self.fds_idxLine["isExactOutputTimeEnforced"]] = "	isExactOutputTimeEnforced = yes\n"
        self.fds_data[self.fds_idxLine["WriteConstraintForceResultFile"]] = "	WriteConstraintForceResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteForceResultFile"]] = "	WriteForceResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteVelocityResultFile"]] = "	WriteVelocityResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteStateResultFile"]] = "	WriteStateResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteAccelerationResultFile"]] = "	WriteAccelerationResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteExtConstraintLagrangeResultFile"]] = "	WriteExtConstraintLagrangeResultFile = no\n"
        self.fds_data[self.fds_idxLine["WriteMeasureResultFile"]] = "	WriteMeasureResultFile = no\n"
        
        self.write_fds(fds)
# -----------------------------------------------------------------------------              

    # def overwrite_param_val_FDS(self):
        
    #     # Open FDS file and store data
    #     with open(self.fds_path_name, 'r') as inp:
    #        self.fds_data = inp.readlines()      
        
    #     tempVar = 0
        
    #     # Get idex of lines
    #     for i, line in enumerate(self.fds_data):
    #         if line.lstrip().startswith("InitialValue"):
    #             self.fds_data[i] = f"	InitialValue = {self.opt_pars[tempVar]}\n"
    #             tempVar = tempVar + 1
    #             if tempVar == 2:
    #                 break
                    
    #     self.write_fds(self.fds_path_name)
# ----------------------------------------------------------------------------- 