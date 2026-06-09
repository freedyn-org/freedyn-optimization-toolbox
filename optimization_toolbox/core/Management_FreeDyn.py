import numpy as np
import freedyn as fd
from ctypes import c_int


class FreeDyn():
    
    def __init__(self,
                 path_FDdll, path_fds, name_fds,
                 name_ctrlSPL, name_fDu_par):
        
        # Initialize FreeDyn API
        fd.initialize(path_FDdll)

        # Define path and name of *.fds
        self.fds_path = path_fds
        self.fds_path_name = f'{path_fds}\\{name_fds}.fds'
        
        # Load and Read *.fds
        self.load_and_read_fds(self.fds_path_name)
        self.fds_set_writing_to_none(self.fds_path_name) # set any file writing of FreeDyn to no
        
        # Create Model
        self.fd_model = fd.Model(self.fds_path_name, status_output="NO")
        info = self.fd_model.get_info()
        self.nDof = info.num_generalized_coordinates
        self.nConstr = info.num_lagrange_multipliers
        self.nDofConstr = self.nDof + self.nConstr 
        self.num_time_steps = 0
        
        # System matrices and derivatives - Decision: dense or sparse layout
        if self.nDof < (10 * 7 + 1):
            self.MBS_modeMAT_sparse = False  
        else:
            self.MBS_modeMAT_sparse = True
            
        # System matrices and derivatives - Decision: dense or sparse layout
        self.init_MBS_SysMat_slots()
        
        # Derivative of sum of external forces w.r.t. parameter given as string
        self.buffer_MBS_fDu = fd.ForceParameterDerivativeMatrixBuffer(name_fDu_par)
        self.fDu = self.buffer_MBS_fDu.data
        
        # FreeDyn data object spline of the controls 
        self.name_ctrlSPL = name_ctrlSPL
        
        print('class FreeDyn initialized')
# -----------------------------------------------------------------------------        
        
    def __del__(self):
        
        self.fd_model.__del__()
        print('Model deleted')   
        
# -----------------------------------------------------------------------------

    def init_MBS_SysMat_slots(self):
        
        # Set up the memory layout either as sparse or dense
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
        
# =============================================================================
# Commands concerning splines
# =============================================================================
    
    def update_ctrl_gridNodes(self):
        
        realT = self.tF * self.ctrl_gridNodes_tau
        
        for i, SPL in enumerate(self.name_ctrlSPL):
            self.fd_model.set_spline(SPL, realT, self.ctrl_gridNodes[:,i])
# -----------------------------------------------------------------------------              
            
    def write_ctrl_dataSPL(self):
        
        realT = self.tF * self.ctrl_gridNodes_tau
        data = np.column_stack((realT, self.ctrl_gridNodes))
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