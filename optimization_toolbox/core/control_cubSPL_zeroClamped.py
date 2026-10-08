import numpy as np

class Control:
    
    def __init__(self, num_ctrls, num_grid_nodes):
        
        self.num_ctrls = num_ctrls
        
        
        if num_ctrls > 0:
            self.num_grid_nodes = num_grid_nodes
            
            self.grid_nodes = None
            
            self.grid_tau = np.linspace(0, 1, num_grid_nodes)
            self.spline_time_invariant()
            print('class Control initialized')

# -----------------------------------------------------------------------------

    def spline_time_invariant(self):
        
        s = self.num_grid_nodes - 1   # number of Splines
        dim_coeffs = 3*s
    
        matA = np.zeros((dim_coeffs, dim_coeffs))
        matK = np.zeros((dim_coeffs, self.num_grid_nodes))
        
        """ vecs with idx from 0 to s-1 | s-2 """
        vec_dim_s = np.arange(s)
        vec_dim_s1 = np.arange(s-1)
        
        """ row idx of Eq. 72 I) II) III) """
        rows_I = 1 + 3 * vec_dim_s
        rows_II = 2 + 3 * vec_dim_s1
        rows_III = 3 + 3 * vec_dim_s1
        
        """ precompute time diffs """
        h = np.diff(self.grid_tau)
        h_pow_2 = h * h
        h_pow_3 = h * h_pow_2
                 
        """ Build matrix A and matrix K """
        # Note: coeffSPL is NOT like in Eq. 71: b_{0} ... b_{s-1}, c_{0} ... c_{s-1}, d_{0} ... d_{s-1}
        #       coeffSPL is collected - b_{0}, c_{0}, d_{0}, ... b_{s-1}, c_{s-1}, d_{s-1}
        
        # Eq 72
        matA[0,0] = 1                                       # BC - Coeffs b_{0}

        # Eq 70 - I
        matA[rows_I, 3 * vec_dim_s] = h                        # Coeffs b_{i}
        matA[rows_I, 3 * vec_dim_s + 1] = h_pow_2              # Coeffs c_{i}
        matA[rows_I, 3 * vec_dim_s + 2] = h_pow_3              # Coeffs d_{i}
        matK[rows_I, vec_dim_s] = 1                            # Coeffs u_{i}
        matK[rows_I, vec_dim_s + 1] = -1                       # Coeffs u_{i+1}
            
        # Eq 70 - II
        matA[rows_II, 3 * vec_dim_s1] = 1                      # Coeffs b_{i}
        matA[rows_II, 3 * vec_dim_s1 + 1] = 2 * h[:-1]         # Coeffs c_{i}
        matA[rows_II, 3 * vec_dim_s1 + 2] = 3 * h_pow_2[:-1]   # Coeffs d_{i}
        matA[rows_II, 3 * vec_dim_s1 + 3] = -1                 # Coeffs b_{i+1}
            
        # Eq 70 - III
        matA[rows_III, 3 * vec_dim_s1 + 1] = 2                 # Coeffs c_{i}
        matA[rows_III, 3 * vec_dim_s1 + 2] = 6 * h[:-1]        # Coeffs d_{i}
        matA[rows_III, 3 * vec_dim_s1 + 4] = -2                # Coeffs c_{i+1}
        
        # Eq 73
        matA[-1, 3 * s - 3] = 1                           # BC - Coeffs b_{s-1}
        matA[-1, 3 * s - 2] = 2 * h[-1]                   # BC - Coeffs c_{s-1}
        matA[-1, 3 * s - 1] = 3 * h_pow_2[-1]             # BC - Coeffs d_{s-1}
        
        # Minus for the c-vector
        self.ctrl_invA_times_K = - np.linalg.solve(matA,matK)  
# -----------------------------------------------------------------------------

    def vec_c_dtF_invariant_OptIt(self, final_time):
        
        inv_tF_squared = 1 / (final_time * final_time)
        self.vecC_dtF_invariant = self.ctrl_invA_times_K @ self.grid_nodes
        self.vecC_dtF_invariant *= inv_tF_squared
        
        self.vecC_dtF_invariant[1::3] *= 2
        self.vecC_dtF_invariant[2::3] *= 3 
# -----------------------------------------------------------------------------

    def find_SPL_by_t(self, t):
        
        idx = np.searchsorted(self.grid_tau, t)
        ctrl_intSPL_pos = min(max(0, idx - 1), self.num_grid_nodes-2)

        return ctrl_intSPL_pos

# -----------------------------------------------------------------------------

    def get_vec_c(self, time_point):
        
        ctrl_intSPL_pos = self.find_SPL_by_t(time_point)
        idx_mat = 3 * ctrl_intSPL_pos
        
        # Vector tau entries
        t = time_point - self.grid_tau[ctrl_intSPL_pos]
        
        
        # Note: ctrl_invA_times_K includes already the minus  
        vec_C = t * ( self.ctrl_invA_times_K[idx_mat, :] 
              + t * ( self.ctrl_invA_times_K[idx_mat + 1, :] 
                + t * self.ctrl_invA_times_K[idx_mat + 2, :]))
            
        vec_C[ctrl_intSPL_pos] += 1.0
            
        return vec_C 
# -----------------------------------------------------------------------------

    def get_vec_c_dtF(self, time_point):
        
        ctrl_intSPL_pos = self.find_SPL_by_t(time_point)
        idx_mat = 3 * ctrl_intSPL_pos
        
        # Vector tau entries
        t = time_point - self.grid_tau[ctrl_intSPL_pos]
        
        # Note: SPL_coeffs_bcd includes already the minus  
        vec_C_dtF =  (self.ctrl_invA_times_K[idx_mat, :] 
                  + t * ( 2 * self.ctrl_invA_times_K[idx_mat + 1, :] 
                  + t * 3 * self.ctrl_invA_times_K[idx_mat + 2, :]))
    
            
        return vec_C_dtF 
# -----------------------------------------------------------------------------

    def get_vec_c_AND_dCdtau_times_uDach(self, time_point):
        
        ctrl_intSPL_pos = self.find_SPL_by_t(time_point)
        idx_mat = 3 * ctrl_intSPL_pos
        
        # Vector tau entries
        t = time_point - self.grid_tau[ctrl_intSPL_pos]
        
        # Note: ctrl_invA_times_K and SPL_coeffs_bcd include already the minus   

        vec_C = t * ( self.ctrl_invA_times_K[idx_mat, :] 
              + t * ( self.ctrl_invA_times_K[idx_mat + 1, :] 
                + t * self.ctrl_invA_times_K[idx_mat + 2, :]))
            
        vec_C[ctrl_intSPL_pos] += 1.0
            
          
        # Note: SPL_coeffs_bcd includes already the minus  
        dCdtau_times_uDach =  (self.vecC_dtF_invariant[idx_mat] 
                      + t * ( self.vecC_dtF_invariant[idx_mat + 1] 
                      +  t * self.vecC_dtF_invariant[idx_mat + 2]))
            
        return vec_C, dCdtau_times_uDach 
# -----------------------------------------------------------------------------

    def get_C(self, time_point):
        
        vec_C = self.get_vec_c(time_point)
        matC = np.zeros([self.num_ctrls,self.num_ctrls*self.num_grid_nodes])
        
        for i in range(0, self.num_ctrls):
             matC[i,i*self.num_grid_nodes:(i+1)*self.num_grid_nodes] = vec_C
          
        return matC
# -----------------------------------------------------------------------------

    def get_C_dtF(self, time_point):
        
        vec_C_dtF = self.get_vec_c_dtF(time_point)

        if self.num_ctrls == 1:
            return vec_C_dtF
        
        else:
            
            matC_dtF = np.zeros([self.num_ctrls,self.num_ctrls*self.num_grid_nodes])
            
            for i in range(0, self.num_ctrls):
                matC_dtF[i,i*self.num_grid_nodes:(i+1)*self.num_grid_nodes] = vec_C_dtF
              
            return matC_dtF
# -----------------------------------------------------------------------------
     
    def get_u(self, time_point):
        
        vec_C = self.get_vec_c(time_point)
        
        return np.dot(vec_C, self.grid_nodes)
# -----------------------------------------------------------------------------
     
    def get_u_for_GridNodes(self, time_point, user_grid_nodes):
        
        if user_grid_nodes.ndim == 1:
            matC = self.get_C(time_point)
            u = matC @ user_grid_nodes
        
        else:
            vec_C = self.get_vec_c(time_point)
            u = np.dot(vec_C, user_grid_nodes)
            
        return u
# -----------------------------------------------------------------------------