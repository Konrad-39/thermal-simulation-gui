import dolfin as df
import numpy as np

class ThermalMesh:
    """Handles mesh creation and adaptive refinement for thermal simulations"""
    
    def __init__(self, parameters, comm=None):
        self.parameters = parameters
        self.comm = comm
        self.mesh = None
        
    def create_mesh(self, comm=None):
        """Create initial mesh based on parameters with optional MPI communicator"""
        L = self.parameters['length']
        W = self.parameters['width'] 
        H = self.parameters['height']
        
        # Calculate optimal resolution
        nx, ny, nz = self.calculate_optimal_mesh_resolution()
        
        if comm is not None:
            self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
        else:
            self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
            
        return self.mesh
        
    def calculate_optimal_mesh_resolution(self):
        """Calculate mesh resolution based on laser spot size"""
        beam_radius = self.parameters['beam_radius']
        beam_diameter = 2 * beam_radius
        
        # At least 8-10 elements across beam diameter
        elements_per_beam = 10
        
        # Calculate required element size
        required_element_size = beam_diameter / elements_per_beam
        
        # Calculate number of elements needed
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        nx = max(20, int(L / required_element_size))
        ny = max(20, int(W / required_element_size))
        nz = max(5, int(H / required_element_size))
        
        # Apply base resolution constraint
        base_res = self.parameters.get('mesh_resolution', 32)
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        
        if scale_factor > 10:
            # For scaled runs, prefer coarser mesh
            nx = min(nx, base_res)
            ny = min(ny, base_res)
        else:
            nx = max(nx, base_res)
            ny = max(ny, base_res)
        
        nz = max(nz, int(base_res * H / L))
        
        print(f"Beam diameter: {beam_diameter*1000:.3f} mm")
        print(f"Required element size: {required_element_size*1000:.3f} mm")
        print(f"Recommended mesh: {nx} x {ny} x {nz}")
        
        return nx, ny, nz

    def get_function_space(self):
        """Get function space for current mesh"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
        return df.FunctionSpace(self.mesh, 'P', 1)

    def calculate_stable_time_step(self):
        """Calculate maximum stable time step for current mesh"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
            
        # Get mesh size
        h_min = self.mesh.hmin()
        
        # Material properties
        k = self.parameters['k']
        rho = self.parameters['rho']
        cp = self.parameters['cp']
        
        # Thermal diffusivity
        alpha = k / (rho * cp)
        
        # Stability condition: dt <= h²/(6*alpha) for 3D
        dt_stable = (h_min**2) / (6 * alpha)
        
        print(f"Minimum element size: {h_min*1000:.4f} mm")
        print(f"Thermal diffusivity: {alpha*1e6:.2f} mm²/s")
        print(f"Maximum stable dt (CFL): {dt_stable:.2e} s")
        
        return dt_stable

    # def adaptive_mesh_refinement(self, u, laser_criteria=None):
    #     """Perform adaptive mesh refinement based on temperature gradients and laser position"""
    #     if not self.parameters.get('adaptive_refinement', False):
    #         return False
    
    #     try:
    #         # Calculate temperature gradient magnitude
    #         DG = df.FunctionSpace(self.mesh, "DG", 0)
    #         grad_u = df.project(df.sqrt(df.dot(df.grad(u), df.grad(u))), DG)
            
    #         # Get refinement parameters
    #         threshold = self.parameters['refinement_threshold'] * grad_u.vector().max()
    #         min_cell_size = self.parameters['min_cell_size']
            
    #         # Get laser info from criteria
    #         if laser_criteria:
    #             laser_x, laser_y = laser_criteria['laser_position']
    #             laser_radius = laser_criteria['laser_radius']
    #             T_melt = laser_criteria['T_melt']
    #         else:
    #             # Fallback values
    #             laser_x = self.parameters['length'] / 2
    #             laser_y = self.parameters['width'] / 2
    #             laser_radius = self.parameters['beam_radius']
    #             T_melt = self.parameters.get('T_melt', 3103.0)

    #         # Mark cells for refinement
    #         cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
    #         cell_markers.set_all(False)
            
    #         cells_to_refine = 0
            
    #         for cell in df.cells(self.mesh):
    #             midpoint = cell.midpoint()
                
    #             # Check if cell is large enough to refine
    #             if cell.h() > min_cell_size:
    #                 # Criterion 1: High temperature gradient
    #                 high_gradient = grad_u(midpoint) > threshold
                    
    #                 # Criterion 2: Near laser spot
    #                 distance = ((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)**0.5
    #                 near_laser = distance < 3 * laser_radius
                    
    #                 # Criterion 3: High temperature (near melting)
    #                 high_temp = False
    #                 try:
    #                     temp_at_point = u(midpoint)
    #                     high_temp = temp_at_point > 0.8 * T_melt
    #                 except:
    #                     high_temp = False
                    
    #                 if high_gradient or near_laser or high_temp:
    #                     cell_markers[cell] = True
    #                     cells_to_refine += 1
            
    #         # Only refine if we have cells to refine
    #         if cells_to_refine > 0:
    #             print(f"Refining {cells_to_refine} cells...")
                
    #             # Store old mesh info
    #             old_num_cells = self.mesh.num_cells()
                
    #             # Refine the mesh
    #             new_mesh = df.refine(self.mesh, cell_markers)
                
    #             # Update mesh reference
    #             self.mesh = new_mesh
                
    #             print(f"Mesh refined: {old_num_cells} -> {new_mesh.num_cells()} cells")
    #             return True
            
    #         return False
            
    #     except Exception as e:
    #         print(f"Warning: Adaptive refinement failed: {e}")
    #         return False

    def adaptive_mesh_refinement(self, u, laser_criteria=None):
        """Perform adaptive mesh refinement based on temperature gradients and laser position"""
        if not self.parameters.get('adaptive_refinement', False):
            return False

        try:
            print("DEBUG: Starting mesh refinement...")
            
            # Calculate temperature gradient magnitude
            print("DEBUG: Creating DG function space...")
            DG = df.FunctionSpace(self.mesh, "DG", 0)
            
            print("DEBUG: Projecting gradient...")
            grad_u = df.project(df.sqrt(df.dot(df.grad(u), df.grad(u))), DG)
            
            print("DEBUG: Getting refinement parameters...")
            threshold = self.parameters['refinement_threshold'] * grad_u.vector().max()
            min_cell_size = self.parameters['min_cell_size']
            
            # Get laser info from criteria
            if laser_criteria:
                laser_x, laser_y = laser_criteria['laser_position']
                laser_radius = laser_criteria['laser_radius']
                T_melt = laser_criteria['T_melt']
            else:
                laser_x = self.parameters['length'] / 2
                laser_y = self.parameters['width'] / 2
                laser_radius = self.parameters['beam_radius']
                T_melt = self.parameters.get('T_melt', 3103.0)

            print("DEBUG: Creating cell markers...")
            cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
            cell_markers.set_all(False)
            
            cells_to_refine = 0
            
            print("DEBUG: Checking cells for refinement...")
            for cell in df.cells(self.mesh):
                midpoint = cell.midpoint()
                
                if cell.h() > min_cell_size:
                    # Simplified criteria to avoid potential issues
                    distance = ((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)**0.5
                    near_laser = distance < 3 * laser_radius
                    
                    if near_laser:  # Only use laser proximity for now
                        cell_markers[cell] = True
                        cells_to_refine += 1
            
            print(f"DEBUG: Found {cells_to_refine} cells to refine")
            
            if cells_to_refine > 0:
                print("DEBUG: About to call df.refine...")
                old_num_cells = self.mesh.num_cells()
                
                # This is likely where the error occurs
                new_mesh = df.refine(self.mesh, cell_markers)
                
                print("DEBUG: df.refine completed successfully")
                self.mesh = new_mesh
                
                print(f"Mesh refined: {old_num_cells} -> {new_mesh.num_cells()} cells")
                return True
            
            return False
            
        except Exception as e:
            print(f"ERROR in mesh refinement at step: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _default_refinement_criteria(self, u, mesh):
        """Default refinement criteria based on temperature gradient"""
        # Calculate gradient magnitude at each cell
        DG = df.FunctionSpace(mesh, "DG", 0)
        grad_u = df.project(df.sqrt(df.dot(df.grad(u), df.grad(u))), DG)
        
        # Return gradient values as refinement indicators
        return grad_u.vector().get_local()