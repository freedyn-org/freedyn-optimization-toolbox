import numpy as np
import freedyn as fd


class FreeDyn():
    
    def __init__(self, 
                 path_fds, name_fds,
                 pathFDdll):
        
        # Initialize FreeDyn API
        fd.initialize(pathFDdll)

        # Define path and name of *.fds
        self.fds_path = path_fds
        self.fds_path_name_original = f'{path_fds}\\{name_fds}.fds'
        self.fds_path_name_optimized = f'{path_fds}\\{name_fds}_optimized.fds'
        
        # Load and Read *.fds
        self.load_and_read_fds(self.fds_path_name_original)
        self.fds_set_writing_to_none(self.fds_path_name_optimized)
        
        # Create Model
        self.fd_model = fd.Model(self.fds_path_name_optimized, status_output="NO")
        info = self.fd_model.get_info()
        self.nDof = info.num_generalized_coordinates
        self.nConstr = info.num_lagrange_multipliers
        self.nDofConstr = self.nDof + self.nConstr 
        self.numTimeSteps = 0
        
        print('class FreeDyn initialized')
        
        
    def __del__(self):
        self.fd_model.__del__()

        print('Model deleted')   
        
# -----------------------------------------------------------------------------
     
    def exec_FreeDyn(self):
        self.fd_model.compute_initial_conditions()
        self.fd_model.solve_until(self.tF) 
        self.numTimeSteps = self.fd_model.get_num_time_steps() 

        return None        

# -----------------------------------------------------------------------------     
    def load_and_read_fds(self, fds):
        
        keys = {"SimulationTimeEnd",
                "OutputTimeStepSize",
                #
                "isExactOutputTimeEnforced",
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
        for i, zeile in enumerate(self.fds_data):
            if zeile.startswith("	SimulationTimeEnd"):
                self.fds_idxLine["SimulationTimeEnd"] = i
            if zeile.startswith("	OutputTimeStepSize"):
                self.fds_idxLine["OutputTimeStepSize"] = i
            if zeile.startswith("	isExactOutputTimeEnforced"):
                self.fds_idxLine["isExactOutputTimeEnforced"] = i
            if zeile.startswith("	WriteConstraintForceResultFile"):
                self.fds_idxLine["WriteConstraintForceResultFile"] = i
            if zeile.startswith("	WriteForceResultFile"):
                self.fds_idxLine["WriteForceResultFile"] = i
            if zeile.startswith("	WriteVelocityResultFile"):
                self.fds_idxLine["WriteVelocityResultFile"] = i
            if zeile.startswith("	WriteStateResultFile"):
                self.fds_idxLine["WriteStateResultFile"] = i
            if zeile.startswith("	WriteAccelerationResultFile"):
                self.fds_idxLine["WriteAccelerationResultFile"] = i
            if zeile.startswith("	WriteExtConstraintLagrangeResultFile"):
                self.fds_idxLine["WriteExtConstraintLagrangeResultFile"] = i
            if zeile.startswith("	WriteMeasureResultFile"):
                self.fds_idxLine["WriteMeasureResultFile"] = i
                
           
        return None

# -----------------------------------------------------------------------------     
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
        
        return None

# -----------------------------------------------------------------------------
    
    def write_fds(self, fds):
        
        with open(fds, 'w') as target:
           target.writelines(self.fds_data)   
           
        return None
                  
# ----------------------------------------------------------------------------   
    
    def change_tF_in_fds(self, fds): 
        self.fds_data[self.fds_tF_idxLine] =  f"	SimulationTimeEnd = {self.tF}\n"
        self.write_fds(fds)
        
        return None
       
# ----------------------------------------------------------------------------   
    
    def change_outputDeltaT_in_fds(self, fds, tau): 
        self.fds_data[self.fds_outputDeltaT_idxLine] =  f"	OutputTimeStepSize = {tau}\n"
        self.write_fds(fds)

        return None
       
# -----------------------------------------------------------------------------

    def write_ctrl_dataSPL(self):
        
        realT = self.tF * self.tDach
        data = np.column_stack((realT, self.uDach))
        np.savetxt(f'{self.fds_path}\\dataSPL.txt', data, fmt='%.10f')
                
        return None
    
# -----------------------------------------------------------------------------          
    
    def update_ctrl_gridNodes(self):
        
        realT = self.tF * self.tDach
        
        for i, SPL in enumerate(self.nameCtrlSpline):
            self.fd_model.set_spline(SPL, realT, self.uDach[:,i])
           
        return None

# ----------------------------------------------------------------------------- 