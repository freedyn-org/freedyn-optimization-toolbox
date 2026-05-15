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
        self.fds_path_name = f'{path_fds}\\{name_fds}.fds'
               
        # Create Model
        self.fd_model = fd.Model(self.fds_path_name, status_output="NO")
        info = self.fd_model.get_info()
        self.nDof = info.num_generalized_coordinates
        self.nConstr = info.num_lagrange_multipliers
        self.nDofConstr = self.nDof + self.nConstr 
        self.dyn_numTimeSteps = 0
        
        print('class FreeDyn initialized')
        
        
    def __del__(self):
        self.fd_model.__del__()

        print('Model deleted')   
        
# -----------------------------------------------------------------------------
     
    def exec_FreeDyn(self):
        self.fd_model.compute_initial_conditions()
        self.fd_model.solve_until(self.tF) 
        self.dyn_numTimeSteps = self.fd_model.get_num_time_steps() 

        return None        

# -----------------------------------------------------------------------------     
    def load_fds(self, fds):
        
        self.fds_tF_idxLine = None
        self.fds_outputDeltaT_idxLine = None
        
        # Open FDS file and store data
        with open(fds, 'r') as inp:
           self.fds_data = inp.readlines()
            
        
        # Get line of final time
        for i, zeile in enumerate(self.fds_data):
            if zeile.startswith("	SimulationTimeEnd ="):
                self.fds_tF_idxLine = i
            if zeile.startswith("	OutputTimeStepSize ="):
                self.fds_outputDeltaT_idxLine = i
           
        return None

# -----------------------------------------------------------------------------
    
    def write_fds(self, fds):
        
        with open(fds, 'w') as target:
           target.writelines(self.fds_data)   
           
        return None
                  
# ----------------------------------------------------------------------------   
    
    def change_tF_in_fds(self): 
        self.fds_data[self.fds_tF_idxLine] =  f"	SimulationTimeEnd = {self.tF}\n"
        self.write_fds()
        
        return None
       
# ----------------------------------------------------------------------------   
    
    def change_outputDeltaT_in_fds(self, tau): 
        self.fds_data[self.fds_outputDeltaT_idxLine] =  f"	OutputTimeStepSize = {tau}\n"
        self.write_fds()

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