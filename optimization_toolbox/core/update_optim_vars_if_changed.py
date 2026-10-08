import numpy as np

    
def update_optim_vars(task):

    if task == "OCP":
        return UpdateOCP()
    elif task == "TOCP":
        return UpdateTOCP()
    elif task == "Parameter":
        return UpdateParam()
    else:
        raise ValueError(f"Optimization task {task} not defined!")
        
# -----------------------------------------------------------------------------       

class UpdateBasic():
    
    def recompute_due_to_change(self, OptimTask, FreeDyn):
        FreeDyn.API.reset_for_rerun()
        FreeDyn.API.compute_initial_conditions()
        FreeDyn.API.solve_until(OptimTask.final_time) 
        FreeDyn.num_time_steps = FreeDyn.API.get_num_time_steps()  
# -----------------------------------------------------------------------------      

class UpdateOCP(UpdateBasic):
    
    def assign_if_changed(self, OptimTask, FreeDyn, Ctrl, z):
        
        """ Check if the solution is already computed for z, otherwise reset and recompute """
        
        # Get new values of z
        mat_ctrl_gridNodes_new = z.reshape((Ctrl.num_grid_nodes, Ctrl.num_ctrls),order='F')              

        # compare of change
        u_changed = not np.array_equal(Ctrl.grid_nodes, mat_ctrl_gridNodes_new)

        # reuse or compute solution, only assign and compute if changed
        if u_changed:
            Ctrl.grid_nodes = mat_ctrl_gridNodes_new.copy()
            FreeDyn.update_ctrl_spline(OptimTask, Ctrl)
            self.recompute_due_to_change(OptimTask, FreeDyn)
            
# -----------------------------------------------------------------------------
    
class UpdateTOCP(UpdateBasic):
    
    def assign_if_changed(self, OptimTask, FreeDyn, Ctrl, z):
        
        """ Check if the solution is already computed for z, otherwise reset and recompute """
        
        # Get new values of z
        new_tf = z[0]
        mat_ctrl_gridNodes_new = z[1:].reshape((Ctrl.num_grid_nodes, Ctrl.num_ctrls),order='F')
        
        # compare of change
        tf_changed = (new_tf != OptimTask.final_time)
        u_changed = not np.array_equal(Ctrl.grid_nodes, mat_ctrl_gridNodes_new)
        
        # reuse or compute solution, only assign and compute if changed
        if tf_changed or u_changed:
            OptimTask.final_time = new_tf
            Ctrl.grid_nodes = mat_ctrl_gridNodes_new.copy()
            FreeDyn.update_ctrl_spline(OptimTask, Ctrl)
            self.recompute_due_to_change(OptimTask, FreeDyn)
    
# -----------------------------------------------------------------------------
    
class UpdateParam(UpdateBasic):
    
    def assign_if_changed(self, OptimTask, FreeDyn, Ctrl, z):
        
        """ Check if the solution is already computed for z, otherwise reset and recompute """
        
        # Get new values of z
        opt_pars_new = z              

        # compare of change
        opt_pars_changed = not np.array_equal(FreeDyn.FD_pars, opt_pars_new)

        # reuse or compute solution, only assign and compute if changed
        if opt_pars_changed:
            FreeDyn.FD_pars = opt_pars_new.copy()
            FreeDyn.update_FD_pars(FreeDyn.name_dForce_dparam, FreeDyn.FD_pars)
            self.recompute_due_to_change(OptimTask, FreeDyn) 
# -----------------------------------------------------------------------------