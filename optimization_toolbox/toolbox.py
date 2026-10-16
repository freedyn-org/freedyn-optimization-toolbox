class Toolbox():
    
    def __init__(self, freedyn, ctrl, optim_task):
        
        self.FreeDyn = freedyn
        self.Ctrl = ctrl
        self.OptimTask = optim_task
        
        print('class Optimization initialized \n')   
# -----------------------------------------------------------------------------

    def new_opt_vars(self, z):
        
        """ Check if solution is already computed for z, 
            otherwise reset + recompute """
        return self.OptimTask.UpdateOptimVars.compute_or_reuse(self.OptimTask, self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------        

    def cost_fct_J(self, z):
        
        """ Cost functional J of the optimization problem
            J = \int_{t_0}^{t_f} L dt """        
        return self.OptimTask.cost_fct_J(self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------
    
    def final_constr_eq_Phi(self, z):
        
        """ Final constraints Phi of the optimization problem
        Phi (t_f) = 0 """        
        return self.OptimTask.final_constr_eq_Phi(self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------

    def grad_cost_fct_J(self, z):
        
        """ Gradient of the cost functional J """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = self.OptimTask.check_grad_J(self.FreeDyn, self.Ctrl, z)

        # Returns the gradient by the adjoint method
        return self.OptimTask.adj_grad_J(self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------

    def grad_final_constr_eq_Phi(self, z):
        
        """ Gradient of the final constraints Phi """
        
        # Verification of the adjoint gradient via numerical differentiation
        # error = self.OptimTask.check_grad_Phi(self.FreeDyn, self.Ctrl, z)
        
        # Returns the gradient by the adjoint method
        return self.OptimTask.adj_grad_Phi(self.FreeDyn, self.Ctrl, z)
# -----------------------------------------------------------------------------