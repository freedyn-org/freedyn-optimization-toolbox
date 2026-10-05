from core.optimization_task import opt_task
from core.Management_FreeDyn import FreeDyn
from core.control_cubSPL_zeroClamped import Control
from core.update_optim_vars_if_changed import update_optVars
import core.numerical_differentiation as numDiff


class Toolbox():
    
    def __init__(self, task,
                      num_optVars, num_ctrls, num_ctrl_gridNodes, num_FD_pars,
                      tF, xF,
                      path_fds, name_fds,
                      name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                      path_FDdll):

        self.ctrl = Control(num_ctrls, num_ctrl_gridNodes)
        self.FreeDyn = FreeDyn(path_FDdll, path_fds, name_fds, 
                               name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                               num_FD_pars)
        self.opt_task = opt_task(task, self.FreeDyn, self.ctrl, 
                                 num_optVars, tF, xF)
        
        self.optim_vars = update_optVars(task)

        print('class Optimization initialized \n')   
# -----------------------------------------------------------------------------
        
    def costFct_J(self, z):
        
        """ Cost functioncal J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
         
        # Check if solution is already computed for z, otherwise reset + recompute
        self.optim_vars.update_optim_vars.check_if_changed(self.opt_task, self.FreeDyn, self.ctrl, z)
        
        return self.opt_task.costFct_J(self.FreeDyn, self.ctrl, z)
# -----------------------------------------------------------------------------
    
    def finalConstr_Phi(self, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.optim_vars.update_optim_vars.check_if_changed(self.opt_task, self.FreeDyn, self.ctrl, z)
        
        return self.opt_task.finalConstr_Phi(self.FreeDyn, z)
# -----------------------------------------------------------------------------

    def grad_costFct_J(self, z):
        
        """ Gradient of the cost functioncal J """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_J(self, z)
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.optim_vars.update_optim_vars.check_if_changed(self.opt_task, self.FreeDyn, self.ctrl, z)
        
        # Returns the gradient by the adjoint method
        return self.opt_task.adjGrads.adjGrad_J(self.opt_task, self.opt_task.user_Fcts, self.FreeDyn, self.ctrl, z)
# -----------------------------------------------------------------------------

    
    def grad_finalConstr_Phi(self, z):
        
        """ Gradient of the final constraints Phi """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_Phi(self, z)
        
        # Check if solution is already computed for z, otherwise reset + recompute
        self.optim_vars.update_optim_vars.check_if_changed(self.opt_task, self.FreeDyn, self.ctrl, z)
        
        # Returns the gradient by the adjoint method
        return self.opt_task.adjGrads.adjGrad_Phi(self.opt_task, self.opt_task.user_Fcts, self.FreeDyn, self.ctrl, z)
# -----------------------------------------------------------------------------