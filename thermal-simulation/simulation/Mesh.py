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



    def plot_mesh_cross_section(self, plane='z', coord=None):
        """Plot mesh cross-section to visualize refinement"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get mesh coordinates
        coordinates = self.mesh.coordinates()
        cells = self.mesh.cells()
        
        # Determine slice coordinate
        if coord is None:
            if plane == 'z':
                coord = self.parameters['height'] / 2
            elif plane == 'y':
                coord = self.parameters['width'] / 2
            elif plane == 'x':
                coord = self.parameters['length'] / 2
        
        # Find cells that intersect the plane
        tol = self.mesh.hmax() * 0.1
        
        for cell_idx, cell in enumerate(cells):
            vertices = coordinates[cell]
            
            # Check if cell intersects plane
            if plane == 'z':
                if min(vertices[:, 2]) <= coord + tol and max(vertices[:, 2]) >= coord - tol:
                    # Project to x-y plane
                    poly = [(v[0]*1000, v[1]*1000) for v in vertices if abs(v[2] - coord) < tol]
            elif plane == 'y':
                if min(vertices[:, 1]) <= coord + tol and max(vertices[:, 1]) >= coord - tol:
                    # Project to x-z plane
                    poly = [(v[0]*1000, v[2]*1000) for v in vertices if abs(v[1] - coord) < tol]
            elif plane == 'x':
                if min(vertices[:, 0]) <= coord + tol and max(vertices[:, 0]) >= coord - tol:
                    # Project to y-z plane
                    poly = [(v[1]*1000, v[2]*1000) for v in vertices if abs(v[0] - coord) < tol]
            
            if len(poly) >= 3:
                polygon = Polygon(poly, fill=False, edgecolor='blue', linewidth=0.5)
                ax.add_patch(polygon)
        
        # Set labels based on plane
        if plane == 'z':
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_title(f'Mesh Cross-Section at Z = {coord*1000:.2f} mm')
        elif plane == 'y':
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Z (mm)')
            ax.set_title(f'Mesh Cross-Section at Y = {coord*1000:.2f} mm')
        elif plane == 'x':
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('Z (mm)')
            ax.set_title(f'Mesh Cross-Section at X = {coord*1000:.2f} mm')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def get_function_space(self):
        """Get function space for current mesh"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
        return df.FunctionSpace(self.mesh, 'P', 1)

    def calculate_stable_time_step(self):
        # """Calculate maximum stable time step for current mesh"""
        # if self.mesh is None:
        #     raise RuntimeError("Mesh not created yet")
            
        # # Get mesh size
        # h_min = self.mesh.hmin()
        # h_max = self.mesh.hmax()
        # h_char = np.sqrt(h_min*h_max)

        # # Material properties
        # k = self.parameters['k']
        # rho = self.parameters['rho']
        # cp = self.parameters['cp']
        
        # # Thermal diffusivity
        # alpha = k / (rho * cp)
        
        # # Stability condition: dt <= h²/(6*alpha) for 3D
        # dt_stable = (h_char**2) / (6 * alpha)
        
        # print(f"Minimum element size: {h_min*1000:.4f} mm")
        # print(f"Thermal diffusivity: {alpha*1e6:.2f} mm²/s")
        # print(f"Maximum stable dt (CFL): {dt_stable:.2e} s")
        
        # return dt_stable
        """Calculate stable time step with consideration for thin samples"""
        
        # Get material properties
        k = self.parameters.get('thermal_conductivity', 45.0)
        rho = self.parameters.get('density', 7850.0)
        cp = self.parameters.get('specific_heat', 460.0)
        
        # Thermal diffusivity
        alpha = k / (rho * cp)
        
        # Get minimum cell size
        h_min = self.mesh.hmin()
        
        # For thin samples, also consider vertical cell size
        thickness = self.parameters['height']
        if thickness < 0.001:  # Thin sample
            # Estimate vertical cell size
            nz_approx = self.mesh.num_cells() ** (1/3) * thickness / self.parameters['length']
            h_z = thickness / max(nz_approx, 10)
            h_min = min(h_min, h_z)
        
        # CFL condition for heat equation
        # dt < h^2 / (2 * alpha) for stability
        safety_factor = 0.25  # Conservative for nonlinear problems
        dt_stable = safety_factor * h_min**2 / (2 * alpha)
        
        return dt_stable


    def create_statisic_optimized_mesh(self, comm=None):
        # """Create optimized mesh once - never changes during simulation"""
        
        # beam_diameter = 2 * self.parameters['beam_radius']  # 1mm
        # thickness = self.parameters['height']               # 0.5mm
        # L = self.parameters['length']                       # 8mm
        # W = self.parameters['width']                        # 8mm
        
        # laser_x = self.parameters.get('laser_start_x', L/2)
        # laser_y = self.parameters.get('laser_start_y', W/2)
        
        # # print("Creating STATIC mesh (no refinement during simulation)")
        # print("Creating MEMORY-EFFICIENT static mesh")
        # print(f"Beam diameter: {beam_diameter*1000:.1f}mm, Sheet thickness: {thickness*1000:.2f}mm")
            
        # # Calculate optimal initial resolution
        # # Core zone: 0.05mm elements (20 across 1mm beam)
        # # Medium zone: 0.1mm elements  
        # # Far zone: 0.2-0.4mm elements
        
        # # Method 1: Direct high-resolution mesh
        # # elements_across_beam = 20
        # # core_element_size = beam_diameter / elements_across_beam  # 0.05mm
        # base_resolution = self.parameters.get('mesh_resolution', 35)

        # # Calculate required base resolution
        # nx =  base_resolution  # Start coarser, will refine zones
        # ny = base_resolution
        # nz = max(4, int(base_resolution * thickness / L))  # 5 layers through 0.5mm thickness = 0.1mm per layer
        
        # print(f"GUI mesh resolution: {base_resolution}")
        # print(f"Base mesh: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
        # print(f"Element size: {L/nx*1000:.2f} x {W/ny*1000:.2f} x {thickness/nz*1000:.2f} mm")
    
        # # Create base mesh
        # if comm is not None:
        #     self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, thickness), 
        #                         nx, ny, nz)
        # else:
        #     self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, thickness), 
        #                         nx, ny, nz)
        
        # print(f"Initial mesh: {self.mesh.num_cells()} cells")
        
        # # Pre-refine zones based on distance from laser (ONCE ONLY)
        # # self._prerefine_zones_static(laser_x, laser_y, beam_diameter)
        # self._conservative_refinement_gui_based(laser_x, laser_y, beam_diameter, base_resolution)

        # print(f"Final STATIC mesh: {self.mesh.num_cells()} cells")
        # print("Mesh will NOT change during simulation")
        
        # return self.mesh

        """Create optimized mesh with through-thickness refinement for thin samples"""
        
        L = self.parameters['length']
        W = self.parameters['width']
        thickness = self.parameters['height']
        
        beam_diameter = 2 * self.parameters['beam_radius']
        laser_x = self.parameters.get('laser_x_position', L/2)
        laser_y = self.parameters.get('laser_y_position', W/2)
        
        base_resolution = self.parameters.get('mesh_resolution', 32)
        
        # Check if sample is thin (< 1mm)
        is_thin_sample = thickness < 0.001  # 1mm
        
        # Calculate base mesh resolution
        nx = base_resolution
        ny = base_resolution
        
        # For thin samples, ensure adequate through-thickness resolution
        if is_thin_sample:
            # Minimum 10 elements through thickness for thin samples
            min_z_elements = 10
            # More elements if sample is very thin
            if thickness < 0.0005:  # 0.5mm
                min_z_elements = 15
            if thickness < 0.0002:  # 0.2mm
                min_z_elements = 20
                
            nz = max(min_z_elements, int(base_resolution * thickness / L))
        else:
            # Standard calculation for thicker samples
            nz = max(4, int(base_resolution * thickness / L))
        
        print(f"Sample thickness: {thickness*1000:.2f} mm")
        print(f"Base mesh: {nx} x {ny} x {nz} = {nx*ny*nz} cells")
        print(f"Element size: {L/nx*1000:.2f} x {W/ny*1000:.2f} x {thickness/nz*1000:.3f} mm")
        
        # Create base mesh
        if comm is not None:
            self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, thickness), 
                                nx, ny, nz)
        else:
            self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, thickness), 
                                nx, ny, nz)
        
        print(f"Initial mesh: {self.mesh.num_cells()} cells")
        
        # Refine with through-thickness consideration
        if is_thin_sample:
            self._refine_thin_sample_mesh(laser_x, laser_y, beam_diameter)
        else:
            self._conservative_refinement_gui_based(laser_x, laser_y, beam_diameter, base_resolution)
        
        print(f"Final STATIC mesh: {self.mesh.num_cells()} cells")
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

    def _refine_thin_sample_mesh(self, laser_x, laser_y, beam_diameter):
        """Special refinement for thin samples with through-thickness consideration"""
        
        beam_radius = beam_diameter / 2
        thickness = self.parameters['height']
        
        # Define refinement zones with vertical extent consideration
        zones = []
        
        if thickness < 0.0005:  # Very thin (< 0.5mm)
            zones = [
                {
                    'radius': beam_radius * 0.5,
                    'refinements': 2,
                    'z_extent': 1.0,  # Refine full thickness
                    'name': 'Core'
                },
                {
                    'radius': beam_radius * 1.5,
                    'refinements': 2,
                    'z_extent': 0.8,  # Refine top 80% of thickness
                    'name': 'Near-field'
                },
                {
                    'radius': beam_radius * 3.0,
                    'refinements': 1,
                    'z_extent': 0.6,  # Refine top 60% of thickness
                    'name': 'Transition'
                }
            ]
        else:  # Moderately thin (0.5-1mm)
            zones = [
                {
                    'radius': beam_radius * 0.6,
                    'refinements': 3,
                    'z_extent': 0.5,  # Refine top half
                    'name': 'Core'
                },
                {
                    'radius': beam_radius * 1.5,
                    'refinements': 2,
                    'z_extent': 0.4,  # Refine top 40%
                    'name': 'Near-field'
                },
                {
                    'radius': beam_radius * 3.0,
                    'refinements': 1,
                    'z_extent': 0.3,  # Refine top 30%
                    'name': 'Transition'
                }
            ]
        
        max_cells = 500000  # Limit for memory
        
        for zone_idx, zone in enumerate(zones):
            if self.mesh.num_cells() > max_cells:
                print(f"Reached cell limit ({max_cells})")
                break
                
            print(f"\nRefining zone: {zone['name']} (r < {zone['radius']*1000:.2f}mm, "
                f"z > {thickness*(1-zone['z_extent'])*1000:.2f}mm)")
            
            for ref_level in range(zone['refinements']):
                if self.mesh.num_cells() > max_cells:
                    break
                    
                cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
                cell_markers.set_all(False)
                
                cells_marked = 0
                for cell in df.cells(self.mesh):
                    midpoint = cell.midpoint()
                    
                    # Check radial distance from laser
                    distance = np.sqrt((midpoint.x() - laser_x)**2 + 
                                    (midpoint.y() - laser_y)**2)
                    
                    # Check if within radial zone AND in upper portion of sample
                    z_threshold = thickness * (1 - zone['z_extent'])
                    
                    if distance < zone['radius'] and midpoint.z() >= z_threshold:
                        cell_markers[cell] = True
                        cells_marked += 1
                
                if cells_marked > 0:
                    print(f"  Level {ref_level+1}: Refining {cells_marked} cells")
                    self.mesh = df.refine(self.mesh, cell_markers)
                    print(f"  New mesh size: {self.mesh.num_cells()} cells")
                else:
                    break

    

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