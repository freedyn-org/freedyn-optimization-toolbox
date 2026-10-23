import numpy as np
import freedyn as fd
from pathlib import Path


class FreeDyn():
    
    def __init__(self,
                 path_FDdll, fds_path, fds_name,
                 fds_ctrl_name, name_fDmeas, fds_dForce_dparam_name,
                 num_FD_pars):
        
        # Initialize FreeDyn API
        fd.initialize(path_FDdll)

        # Define path and name of *.fds
        self.fds_path = fds_path
        self.fds_path_name = str(Path(fds_path) / f'{fds_name}.fds')
       
        
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
        self.buffer_MBS_dForce_dFDparam = fd.ForceParameterDerivativeMatrixBuffer(fds_dForce_dparam_name)
        self.dForce_dFDparam = self.buffer_MBS_dForce_dFDparam.data
        
        # FreeDyn data object control splines, measures, force wrt parameter
        self.fds_ctrl_name = fds_ctrl_name
        self.name_meas = name_fDmeas
        self.fds_dForce_dparam_name = fds_dForce_dparam_name
        self.FD_pars = None
        self.num_FD_pars = num_FD_pars
        
        print('class FreeDyn initialized')
# -----------------------------------------------------------------------------        
        
    def delete_model(self):
        
        self.API.__del__()
        print('Model deleted')

# =============================================================================
# Batch of commands
# =============================================================================        
        
    def fetch_and_update_states_at_index(self, idx):
        self.API.fetch_states_at_index(idx)
        self.API.update_state_at_index(idx) # necessary, if measures are used in get_lagrangian()
# -----------------------------------------------------------------------------

    def compute_due_to_change(self, final_time):
        self.API.reset_for_rerun()
        self.API.compute_initial_conditions()
        self.API.solve_until(final_time) 
        self.num_time_steps = self.API.get_num_time_steps()

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
        
        for i, SPL in enumerate(self.fds_ctrl_name):
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
        
        # Open fds file and store data
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

    # def overwrite_param_val_fds(self):
        
    #     # Open fds file and store data
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
# ----------------------------------------------------------------------------- name