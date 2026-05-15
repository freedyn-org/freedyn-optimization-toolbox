import numpy as np
import freedyn as fd
from ctypes import c_int

from class_Manager_MBS_SysMat_Deriv import MBS_SysMat 

class FreeDyn(MBS_SysMat):
    
    def __init__(self, 
                 path_fds, name_fds,
                 nameCtrlSpline, nameParFdu,
                 pathFDdll):
        
        # Initialize FreeDyn API
        fd.initialize(pathFDdll)

        # Define path and name of *.fds - without file typ!
        # Inhalt in neue Datei schreiben (überschreibt, falls bereits vorhanden)
        self.fds_path = path_fds
        self.fds_real = f'{path_fds}\\{name_fds}.fds'
        self.fds_opt = f'{path_fds}\\{name_fds}_modify_optim.fds'
        
        # Create *.fds brother for changing tF
        self.load_fds()
        self.change_tF_in_fds()
        
        
        self.fd_model = fd.Model(self.fds_opt, status_output="NO")
        info = self.fd_model.get_info()
        self.nDof = info.num_generalized_coordinates
        self.nConstr = info.num_lagrange_multipliers
        self.nDofConstr = self.nDof + self.nConstr 
        
        # get length of DOFs and Constraints
        self.dyn_numTimeSteps = 0

        self.create_identy_matMBS()
        self.create_ID_MBS()
        
        # Derivative of sum of external forces w.r.t. parameter given as string
        self.nameParFdu = nameParFdu
        MBS_SysMat.__init__(self)
        
        self.nameCtrlSpline = nameCtrlSpline
        
                
        print('class FreeDyn initialized')
        
        
    def __del__(self):
        self.fd_model.__del__()

        print('Model deleted')   
        
# -----------------------------------------------------------------------------
     
    def exec_FreeDyn(self):
        
        self.fd_model.__del__()
        
        self.fd_model = fd.Model(self.fds_opt, status_output="NO")
        self.create_ID_MBS()
        self.update_MBS_SysMat_idx()
        
        # if self.FDstoresSOL:    
        #self.fd_model.reset_for_rerun()
        
        #self.update_ctrl_gridNodes()
        
        # Simulate the model 
        self.fd_model.compute_initial_conditions()
        self.fd_model.solve_until(self.tF) 
        self.dyn_numTimeSteps = self.fd_model.get_num_time_steps() 

        return None        

# -----------------------------------------------------------------------------     
    def load_fds(self):
        
        self.fds_tF_idxLine = None
        self.fds_outputDeltaT_idxLine = None
        
        # Open FDS file and store data
        with open(self.fds_real, 'r') as inp:
           self.fds_data = inp.readlines()
            
        
        # Get line of final time
        for i, zeile in enumerate(self.fds_data):
            if zeile.startswith("	SimulationTimeEnd ="):
                self.fds_tF_idxLine = i
            if zeile.startswith("	OutputTimeStepSize ="):
                self.fds_outputDeltaT_idxLine = i
           
        return None

# -----------------------------------------------------------------------------
    
    def write_fds(self):
        
        with open(self.fds_opt, 'w') as target:
           target.writelines(self.fds_data)   
           
        return None
                  
# ----------------------------------------------------------------------------   
    
    def change_tF_in_fds(self): 

        self.fds_data[self.fds_tF_idxLine] =  f"	SimulationTimeEnd = {self.tF}\n"
        
        self.write_fds()
        
        self.dyn_numTimeSteps = 0
        
        return None
       
# ----------------------------------------------------------------------------   
    
    def change_outputDeltaT_in_fds(self, tau): 

        self.fds_data[self.fds_outputDeltaT_idxLine] =  f"	OutputTimeStepSize = {tau}\n"
        
        self.write_fds()
        
        self.dyn_numTimeSteps = 0
        
        return None
       
# -----------------------------------------------------------------------------

    def write_ctrl_dataSPL(self):
        
        realT = self.tF * self.tDach
        data = np.column_stack((realT, self.uDach))
        
        np.savetxt(f'{self.fds_path}\\dataSPL.txt', data, fmt='%.10f')
        
        self.dyn_numTimeSteps = 0
                
        return None
    
# -----------------------------------------------------------------------------          
    
    def update_ctrl_gridNodes(self):
        
        realT = self.tF * self.tDach
        
        for i, SPL in enumerate(self.nameCtrlSpline):
            self.fd_model.set_spline(SPL, realT, self.uDach[:,i])
           
        return None

# -----------------------------------------------------------------------------

    def create_identy_matMBS(self):
        
        self.MBS_singleMat_row = np.array([0], dtype=c_int)
        self.MBS_singleMat_col = np.array([0], dtype=c_int)
        self.MBS_singleMat_scale = np.array([1.0])
        
        self.identy_MBS_MASS = np.array([101], dtype=c_int)     # "MASS"
        self.identy_MBS_CQ = np.array([301], dtype=c_int)          # "CQ"
        self.identy_MBS_CQDT = np.array([302], dtype=c_int)        # "CQDT"
        self.identy_MBS_DQEXTDQD = np.array([109], dtype=c_int)    # "DQEXTDQD"
        
        
        # G^T = "DQEXTDQ" - "DCQTLEDQ" - "DCQTLIDQ" - "DMQDDDQ"
        self.MBS_G_tr_row = np.array([0, 0, 0, 0], dtype=c_int)
        self.MBS_G_tr_col = np.array([0, 0, 0, 0], dtype=c_int)
        self.MBS_G_tr_scale = np.array([1.0, -1.0, -1.0, -1.0])
        self.identy_MBS_G_tr = np.array([108, 110, 105, 102], dtype=c_int)

        return None


# -----------------------------------------------------------------------------
        
    def create_ID_MBS(self):

        # Mass matrix M
        self.ID_M = fd.analysis.create_matrix(self.identy_MBS_MASS, 
                                              self.MBS_singleMat_row, 
                                              self.MBS_singleMat_col, 
                                              self.MBS_singleMat_scale) 
        
        # Constraint Jacobian Cq
        self.ID_Cq = fd.analysis.create_matrix(self.identy_MBS_CQ, 
                                               self.MBS_singleMat_row, 
                                               self.MBS_singleMat_col, 
                                               self.MBS_singleMat_scale)
        
        # MBS_CQDT
        self.ID_CqvDq = fd.analysis.create_matrix(self.identy_MBS_CQDT, 
                                                  self.MBS_singleMat_row, 
                                                  self.MBS_singleMat_col, 
                                                  self.MBS_singleMat_scale)
        
        # fv
        self.ID_fv = fd.analysis.create_matrix(self.identy_MBS_DQEXTDQD, 
                                               self.MBS_singleMat_row, 
                                               self.MBS_singleMat_col, 
                                               self.MBS_singleMat_scale)     
        
        # mat G^T = fq - CqTxlaDq_e - CqTxlaDq_i - MxqddDq
        self.ID_G = fd.analysis.create_matrix(self.identy_MBS_G_tr, 
                                              self.MBS_G_tr_row, 
                                              self.MBS_G_tr_col, 
                                              self.MBS_G_tr_scale)
        
        return None
    
# -----------------------------------------------------------------------------