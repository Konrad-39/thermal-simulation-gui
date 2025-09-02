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
        h_max = self.mesh.hmax()
        h_char = np.sqrt(h_min*h_max)

        # Material properties
        k = self.parameters['k']
        rho = self.parameters['rho']
        cp = self.parameters['cp']
        
        # Thermal diffusivity
        alpha = k / (rho * cp)
        
        # Stability condition: dt <= h²/(6*alpha) for 3D
        dt_stable = (h_char**2) / (6 * alpha)
        
        print(f"Minimum element size: {h_min*1000:.4f} mm")
        print(f"Thermal diffusivity: {alpha*1e6:.2f} mm²/s")
        print(f"Maximum stable dt (CFL): {dt_stable:.2e} s")
        
        return dt_stable

    def create_statisic_optimized_mesh(self, comm=None):
        """Create optimized mesh once - never changes during simulation"""
        
        beam_diameter = 2 * self.parameters['beam_radius']  # 1mm
        thickness = self.parameters['height']               # 0.5mm
        L = self.parameters['length']                       # 8mm
        W = self.parameters['width']                        # 8mm
        
        laser_x = self.parameters.get('laser_start_x', L/2)
        laser_y = self.parameters.get('laser_start_y', W/2)
        
        # print("Creating STATIC mesh (no refinement during simulation)")
        print("Creating MEMORY-EFFICIENT static mesh")
        print(f"Beam diameter: {beam_diameter*1000:.1f}mm, Sheet thickness: {thickness*1000:.2f}mm")
            
        # Calculate optimal initial resolution
        # Core zone: 0.05mm elements (20 across 1mm beam)
        # Medium zone: 0.1mm elements  
        # Far zone: 0.2-0.4mm elements
        
        # Method 1: Direct high-resolution mesh
        # elements_across_beam = 20
        # core_element_size = beam_diameter / elements_across_beam  # 0.05mm
        base_resolution = self.parameters.get('mesh_resolution', 35)

        # Calculate required base resolution
        nx =  base_resolution  # Start coarser, will refine zones
        ny = base_resolution
        nz = max(4, int(base_resolution * thickness / L))  # 5 layers through 0.5mm thickness = 0.1mm per layer
        
        print(f"GUI mesh resolution: {base_resolution}")
        print(f"Base mesh: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
        print(f"Element size: {L/nx*1000:.2f} x {W/ny*1000:.2f} x {thickness/nz*1000:.2f} mm")
    
        # Create base mesh
        if comm is not None:
            self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, thickness), 
                                nx, ny, nz)
        else:
            self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, thickness), 
                                nx, ny, nz)
        
        print(f"Initial mesh: {self.mesh.num_cells()} cells")
        
        # Pre-refine zones based on distance from laser (ONCE ONLY)
        # self._prerefine_zones_static(laser_x, laser_y, beam_diameter)
        self._conservative_refinement_gui_based(laser_x, laser_y, beam_diameter, base_resolution)

        print(f"Final STATIC mesh: {self.mesh.num_cells()} cells")
        print("Mesh will NOT change during simulation")
        
        return self.mesh

    def _prerefine_zones_static(self, laser_x, laser_y, beam_diameter):
        """Pre-refine zones once - mesh then stays fixed"""
        
        beam_radius = beam_diameter / 2
        
        # Zone definitions for 1mm beam
        zones = [
            {'radius': 0.3e-3, 'refinements': 3},  # Ultra-fine core
            {'radius': 0.7e-3, 'refinements': 2},  # Fine near-field
            {'radius': 1.5e-3, 'refinements': 2},  # Medium field
            {'radius': 3.0e-3, 'refinements': 1},  # Transition zone
        ]
        
        total_refinements = 0
        
        for zone_idx, zone in enumerate(zones):
            print(f"\nPre-refining zone {zone_idx+1}: r < {zone['radius']*1000:.1f}mm")
            
            for ref_level in range(zone['refinements']):
                cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                cell_markers.set_all(False)
                
                cells_marked = 0
                
                for cell in df.cells(self.mesh):
                    midpoint = cell.midpoint()
                    
                    distance = np.sqrt((midpoint.x() - laser_x)**2 + 
                                    (midpoint.y() - laser_y)**2)
                    
                    if distance < zone['radius']:
                        cell_markers[cell] = True
                        cells_marked += 1
                
                if cells_marked > 0:
                    print(f"  Level {ref_level+1}: refining {cells_marked} cells")
                    self.mesh = df.refine(self.mesh, cell_markers)
                    total_refinements += 1
                else:
                    print(f"  Level {ref_level+1}: no cells to refine")
                    break
                
                # Safety check
                if self.mesh.num_cells() > 200000:
                    print(f"  Stopping: mesh has {self.mesh.num_cells()} cells")
                    return
        
        print(f"\nCompleted {total_refinements} total refinements")

    def create_direct_high_res_mesh(self, comm=None):
        """Create high-resolution mesh directly - no refinement at all"""
        
        L = self.parameters['length']     # 8mm
        W = self.parameters['width']      # 8mm  
        H = self.parameters['height']     # 0.5mm
        beam_radius = self.parameters['beam_radius']  # 0.5mm
        
        # Target: 0.1mm elements in center, 0.4mm at edges
        # For 1mm beam, use:
        nx = 60  # 8mm / 60 = 0.133mm average (finer near center)
        ny = 60  # 8mm / 60 = 0.133mm average  
        nz = 5   # 0.5mm / 5 = 0.1mm layers
        
        print(f"Creating direct high-res mesh: {nx} x {ny} x {nz}")
        print(f"Average element size: {L/nx*1000:.2f} x {W/ny*1000:.2f} x {H/nz*1000:.2f} mm")
        
        if comm is not None:
            self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
        else:
            self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
        
        print(f"Final mesh: {self.mesh.num_cells()} cells (STATIC)")
        return self.mesh

    def _conservative_refinement_gui_based(self, laser_x, laser_y, beam_diameter, gui_resolution):
        """Apply refinement scaled to GUI resolution"""
        
        beam_radius = beam_diameter / 2
        
        # Scale refinement based on GUI resolution
        if gui_resolution <= 20:
            # Coarse GUI setting - minimal refinement
            zones = [
                {'radius': 0.5e-3, 'refinements': 1},  # Just one level
            ]
            max_cells = 20000
        elif gui_resolution <= 35:
            # Medium GUI setting - moderate refinement  
            zones = [
                {'radius': 0.4e-3, 'refinements': 2},  # Core
                {'radius': 1.0e-3, 'refinements': 1},  # Near field
            ]
            max_cells = 50000
        else:
            # Fine GUI setting - more refinement
            zones = [
                {'radius': 0.3e-3, 'refinements': 2},  # Core
                {'radius': 0.7e-3, 'refinements': 1},  # Near field  
                {'radius': 1.5e-3, 'refinements': 1},  # Medium field
            ]
            max_cells = 80000
        
        print(f"GUI resolution {gui_resolution} -> max {max_cells} cells")
        
        # Apply refinement with memory limit
        for zone_idx, zone in enumerate(zones):
            if self.mesh.num_cells() > max_cells:
                print(f"Stopping refinement: {self.mesh.num_cells()} cells exceeds limit {max_cells}")
                break
                
            print(f"Refining zone {zone_idx+1}: r < {zone['radius']*1000:.1f}mm")
            
            for ref_level in range(zone['refinements']):
                if self.mesh.num_cells() > max_cells:
                    print(f"  Stopping at {self.mesh.num_cells()} cells")
                    break
                    
                cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                cell_markers.set_all(False)
                
                cells_marked = 0
                for cell in df.cells(self.mesh):
                    midpoint = cell.midpoint()
                    distance = np.sqrt((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)
                    
                    if distance < zone['radius']:
                        cell_markers[cell] = True
                        cells_marked += 1
                
                if cells_marked > 0:
                    print(f"  Refining {cells_marked} cells")
                    self.mesh = df.refine(self.mesh, cell_markers)
                    print(f"  New mesh size: {self.mesh.num_cells()} cells")
                else:
                    print(f"  No cells to refine in zone {zone_idx+1}")
                    break