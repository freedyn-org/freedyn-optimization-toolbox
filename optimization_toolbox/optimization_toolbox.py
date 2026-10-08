from core.optimization_task import OptimTask
from core.Management_FreeDyn import FreeDyn
from core.control_cubSPL_zeroClamped import Control
from core.update_optim_vars_if_changed import update_optim_vars
import core.numerical_differentiation as numDiff


class Toolbox():
    
    def __init__(self, task,
                      num_optVars, num_ctrls, num_ctrl_gridNodes, num_FD_pars,
                      tF, xF,
                      path_fds, name_fds,
                      name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                      path_FDdll):
        
        self.UpdateOptimVars = update_optim_vars(task)
        self.FreeDyn = FreeDyn(path_FDdll, path_fds, name_fds, 
                               name_ctrlSPL, name_fDmeas, name_dForce_dparam,
                               num_FD_pars)
        self.Ctrl = Control(num_ctrls, num_ctrl_gridNodes)
        self.OptimTask = OptimTask(task, self.FreeDyn, self.Ctrl, 
                                 num_optVars, tF, xF)
        

        print('class Optimization initialized \n')   
# -----------------------------------------------------------------------------

    def new_opt_vars(self, z):
        
        """ Check if solution is already computed for z, 
            otherwise reset + recompute """
        return self.UpdateOptimVars.assign_if_changed(self.OptimTask, self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------        

    def cost_fct_J(self, z):
        
        """ Cost functional J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """
         
        # Compute or reuse solution for z
        self.new_opt_vars(z)
        
        return self.OptimTask.cost_fct_J(self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------
    
    def final_constr_eq_Phi(self, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """
        
        # Compute or reuse solution for z
        self.new_opt_vars(z)
        
        return self.OptimTask.final_constr_eq_Phi(self.FreeDyn, z)
# -----------------------------------------------------------------------------

    def grad_cost_fct_J(self, z):
        
        """ Gradient of the cost functional J """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_J(self, z)
        
        # Compute or reuse solution for z
        self.new_opt_vars(z)
        
        # Returns the gradient by the adjoint method
        return self.OptimTask.adjGrads.adjGrad_J(self.OptimTask, self.OptimTask.UserFcts, self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------

    def grad_final_constr_eq_Phi(self, z):
        
        """ Gradient of the final constraints Phi """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = numDiff.check_grad_Phi(self, z)
        
        # Compute or reuse solution for z
        self.new_opt_vars(z)
        
        # Returns the gradient by the adjoint method
        return self.OptimTask.adjGrads.adjGrad_Phi(self.OptimTask, self.OptimTask.UserFcts, self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------