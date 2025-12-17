import dolfin as df
import numpy as np


class ThermalMesh1D:
    """Simplified version of ThermalMesh for 1D problems"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        
    def create_graded_mesh_periodic(self):
        """Create graded mesh optimized for periodic BC"""
        L = self.parameters['length']
        k = self.parameters['k']
        rho = self.parameters['rho']
        cp = self.parameters['cp']
        omega = self.parameters['omega']
        
        # Calculate key length scales
        alpha = k / (rho * cp)
        penetration_depth = np.sqrt(2 * alpha / omega)
        
        # Determine mesh parameters
        cells_in_boundary_layer = 20  # Minimum cells in penetration depth
        min_cell_size = penetration_depth / cells_in_boundary_layer
        
        # Use exponential stretching
        n_cells = int(self.parameters.get('mesh_resolution', 100))
        stretching_factor = 3.0  # How much to stretch
        
        # Generate points with exponential distribution
        xi = np.linspace(0, 1, n_cells + 1)
        x_points = L * (np.exp(stretching_factor * xi) - 1) / (np.exp(stretching_factor) - 1)
        
        # Ensure minimum resolution near boundary
        # Refine near x=0 if needed
        refined_points = [0.0]
        for i in range(1, len(x_points)):
            if x_points[i] < 3 * penetration_depth:
                # Add intermediate points if spacing is too large
                if x_points[i] - refined_points[-1] > min_cell_size:
                    n_intermediate = int((x_points[i] - refined_points[-1]) / min_cell_size)
                    for j in range(1, n_intermediate):
                        refined_points.append(refined_points[-1] + min_cell_size)
            refined_points.append(x_points[i])
        
        x_points = np.unique(np.array(refined_points))
        
        # Create mesh with custom points using MeshEditor
        mesh = df.Mesh()
        editor = df.MeshEditor()
        editor.open(mesh, "interval", 1, 1)
        
        n_vertices = len(x_points)
        n_cells = n_vertices - 1
        
        editor.init_vertices(n_vertices)
        editor.init_cells(n_cells)
        
        # Add vertices - FIXED: pass as list
        for i in range(n_vertices):
            editor.add_vertex(i, [float(x_points[i])])  # Pass as list with one element
        
        # Add cells (connecting consecutive vertices)
        for i in range(n_cells):
            editor.add_cell(i, [i, i+1])  # Can pass as regular list
        
        editor.close()
        
        # Print mesh info
        print(f"  Created graded mesh with {n_cells} cells")
        print(f"  Penetration depth: {penetration_depth*1000:.1f} mm")
        print(f"  Min cell size: {min_cell_size*1000:.3f} mm")
        
        return mesh


class ThermalMesh:
    """Handles mesh creation and adaptive refinement for thermal simulations"""
    
    def __init__(self, parameters, comm=None):
        self.parameters = parameters
        self.comm = comm
        self.mesh = None
        
    def create_mesh(self, comm=None):
        """Create initial mesh optimized for AMR"""
        if self.parameters.get('use_graded_mesh', True):
            return self.create_graded_mesh(comm)
        else:
            return self._create_uniform_mesh(comm)
        # L = self.parameters['length']
        # W = self.parameters['width'] 
        # H = self.parameters['height']
        
        # # Create coarse initial mesh for AMR
        # nx, ny, nz = self._calculate_initial_resolution()
        
        # if comm is not None:
        #     self.mesh = df.BoxMesh(comm, df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
        # else:
        #     self.mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, H), nx, ny, nz)
        
        # # Apply initial refinement near laser
        # self._apply_initial_refinement()

        #     # Calculate maximum allowed cell size based on diffusion
        # dt_target = self.parameters.get('dt', 0.001)
        # alpha = self.parameters.get('thermal_conductivity', 45.0) / \
        #         (self.parameters.get('density', 7850.0) * self.parameters.get('specific_heat', 460.0))
        
        # # Maximum cell size for stable diffusion
        # h_max_allowed = np.sqrt(4 * alpha * dt_target)  # With safety factor
        
        # # Ensure initial mesh respects this
        # min_divisions = int(np.ceil(max(L, W, H) / h_max_allowed))
        
        # nx = max(nx, min_divisions)
        # ny = max(ny, min_divisions)
        # nz = max(nz, int(min_divisions * H / max(L, W)))  # Proportional in Z
        
        # print(f"Mesh divisions adjusted for diffusion: {nx}×{ny}×{nz}")
        
        # return self.mesh
    
    def create_uniform_mesh(self, comm=None):
        """Create uniform mesh (old method for compatibility)"""
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        # Use fixed divisions based on beam size
        beam_radius = self.parameters['beam_radius']
        cells_across_beam = self.parameters.get('cells_across_beam', 4)
        
        cell_size = beam_radius / cells_across_beam
        nx = int(np.ceil(L / cell_size))
        ny = int(np.ceil(W / cell_size))
        
        # Z-resolution
        z_res_microns = self.parameters.get('z_resolution_microns', 1.0)
        z_res = z_res_microns * 1e-6
        nz = int(np.ceil(H / z_res))
        
        print(f"Creating uniform mesh: {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
        
        return BoxMesh(comm, Point(0, 0, 0), Point(L, W, H), nx, ny, nz)
    # Add this method to the ThermalMesh class in Mesh.py
    def should_adapt_mesh(self, step, time, amr_interval):
        """Check if mesh adaptation should occur at this step"""
        # Don't adapt at step 0
        if step == 0:
            return False
        
        # Check interval
        if step % amr_interval != 0:
            return False
        
        # Use internal _should_adapt for additional checks
        return self._should_adapt(self.mesh, time)

    def conservative_interpolation(self, u_old, mesh_old, u_new, mesh_new):
        """Interpolate while conserving total thermal energy"""
        # Calculate total energy on old mesh
        mass_old = df.assemble(u_old * df.dx)
        
        # Standard interpolation
        u_new.interpolate(u_old)
        
        # Calculate total energy on new mesh
        mass_new = df.assemble(u_new * df.dx)
        
        # Correct for conservation
        if mass_new > 0:
            correction_factor = mass_old / mass_new
            u_new.vector()[:] *= correction_factor

    def calculate_stable_time_step(self):
        """Calculate stable time step based on largest cell size"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
        
        # Material properties (account for temperature dependence if enabled)
        if hasattr(self, 'parameters') and self.parameters.get('use_temperature_dependent_properties', False):
            # Use worst-case (highest diffusivity)
            k_max = self.parameters.get('k', 45.0) * 2  # Assume k could double
            rho_min = self.parameters.get('rho', 7850.0) * 0.8  # Assume rho could decrease
            cp_min = self.parameters.get('cp', 460.0) * 0.8  # Assume cp could decrease
            alpha_max = k_max / (rho_min * cp_min)
        else:
            k = self.parameters.get('thermal_conductivity', 45.0)
            rho = self.parameters.get('density', 7850.0)
            cp = self.parameters.get('specific_heat', 460.0)
            alpha_max = k / (rho * cp)
        
        # Maximum cell size (largest cell in mesh)
        h_max = self.mesh.hmax()
        
        # CFL condition for heat equation: dt < h²/(2*α)
        # Using h_max ensures stability for ALL cells
        safety_factor = 0.25
        dt_stable_max_cell = safety_factor * h_max**2 / (2 * alpha_max)
        
        # Also check minimum cell for accuracy
        h_min = self.mesh.hmin()
        dt_stable_min_cell = safety_factor * h_min**2 / (2 * alpha_max)
        
        print(f"  Mesh cell sizes: min={h_min*1000:.3f}mm, max={h_max*1000:.3f}mm")
        print(f"  Stable dt for largest cell: {dt_stable_max_cell:.2e}s")
        print(f"  Stable dt for smallest cell: {dt_stable_min_cell:.2e}s")
        
        return dt_stable_max_cell

    def check_diffusion_length(self, dt):
        """Check if heat can diffuse across largest cell in one timestep"""
        if self.mesh is None:
            return False, "No mesh"
        
        # Get thermal diffusivity
        k = self.parameters.get('thermal_conductivity', 45.0)
        rho = self.parameters.get('density', 7850.0)
        cp = self.parameters.get('specific_heat', 460.0)
        alpha = k / (rho * cp)
        
        # Diffusion length in time dt
        diffusion_length = np.sqrt(2 * alpha * dt)
        
        # Maximum cell size
        h_max = self.mesh.hmax()
        
        # Check if diffusion length covers the cell
        ratio = diffusion_length / h_max
        
        is_ok = ratio >= 0.5  # Want diffusion to cover at least half the cell
        
        msg = f"Diffusion length: {diffusion_length*1000:.3f}mm, Max cell: {h_max*1000:.3f}mm, Ratio: {ratio:.2f}"
        
        return is_ok, msg

    def should_adapt_based_on_change(self, temp_change_rate, gradient_magnitude):
        """Adapt when solution is relatively stable"""
        # Get thresholds from parameters or use defaults
        temp_change_threshold = self.parameters.get('temp_change_threshold_for_adaptation', 100.0)  # K/s
        gradient_threshold = self.parameters.get('gradient_threshold_for_adaptation', 1000.0)  # K/m
        
        # Don't adapt during rapid temperature changes
        if temp_change_rate > temp_change_threshold:
            print(f"  Skipping adaptation: temp change rate {temp_change_rate:.1f} K/s > {temp_change_threshold} K/s")
            return False
        
        # Only adapt if gradients are significant but stable
        if gradient_magnitude > gradient_threshold:
            print(f"  Adaptation triggered: gradient {gradient_magnitude:.1f} K/m > {gradient_threshold} K/m")
            return True
            
        return False

    def create_graded_mesh(self, comm=None):
        """Create a graded mesh with fine resolution at laser spot and through thickness"""
        import numpy as np
        from dolfin import Point, BoxMesh, MeshFunction, refine
        
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        beam_radius = self.parameters['beam_radius']
        
        # Laser position
        laser_x = self.parameters.get('laser_x_position', L/2)
        laser_y = self.parameters.get('laser_y_position', W/2)
        
        # Target resolutions
        cells_across_beam = 4
        target_cell_size_at_beam = beam_radius / cells_across_beam  # ~0.125mm for 0.5mm beam
        z_resolution = 1e-6  # 1 micron
        
        # Calculate diffusion length at domain edge (for coarsest cells)
        dt = self.parameters.get('dt', 0.001)
        alpha = self.parameters['k'] / (self.parameters['rho'] * self.parameters['cp'])
        diffusion_length = np.sqrt(2 * alpha * dt)
        max_cell_size = min(diffusion_length, min(L, W) / 10)  # Don't exceed 1/10 domain size
        
        print(f"\nCreating graded mesh:")
        print(f"  Target cell size at beam: {target_cell_size_at_beam*1000:.3f} mm")
        print(f"  Max cell size at edges: {max_cell_size*1000:.3f} mm")
        print(f"  Z-resolution: {z_resolution*1e6:.1f} μm")
        
        # Calculate number of Z-layers for 1 micron resolution
        nz = int(np.ceil(H / z_resolution))
        if nz > 1000:  # Practical limit
            print(f"  WARNING: {nz} Z-layers needed for 1μm resolution. Limiting to 1000.")
            nz = min(1000, nz)
            actual_z_res = H / nz
            print(f"  Actual Z-resolution: {actual_z_res*1e6:.1f} μm")
        
        # Start with a base mesh that captures the beam
        nx_base = int(np.ceil(L / target_cell_size_at_beam / 2))  # Start moderate
        ny_base = int(np.ceil(W / target_cell_size_at_beam / 2))
        
        print(f"  Base mesh: {nx_base} x {ny_base} x {nz}")
        
        # Create base mesh
        mesh = BoxMesh(comm, Point(0, 0, 0), Point(L, W, H), nx_base, ny_base, nz)
        
        # Now apply graded refinement
        mesh = self._apply_graded_refinement(mesh, laser_x, laser_y, beam_radius, 
                                            target_cell_size_at_beam, max_cell_size)
        
        return mesh

    def _apply_graded_refinement(self, mesh, laser_x, laser_y, beam_radius, 
                                min_cell_size, max_cell_size):
        """Apply graded refinement - fine at laser, coarse at edges"""
        import dolfin as df
        
        refinement_levels = 3  # Number of refinement zones
        
        for level in range(refinement_levels):
            cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
            cell_markers.set_all(False)
            
            cells_marked = 0
            for cell in df.cells(mesh):
                # Get cell center and size
                midpoint = cell.midpoint()
                cell_size = cell.h()
                
                # Distance from laser center (in XY plane)
                dist = np.sqrt((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)
                
                # Determine target cell size based on distance
                if dist < beam_radius * 2:
                    # Fine zone: near laser
                    target_size = min_cell_size
                elif dist < beam_radius * 5:
                    # Medium zone: transition
                    target_size = min_cell_size * 2
                else:
                    # Coarse zone: far field
                    # Linear interpolation to max_cell_size at domain edge
                    max_dist = np.sqrt((self.parameters['length']/2)**2 + 
                                    (self.parameters['width']/2)**2)
                    t = min(1.0, (dist - beam_radius * 5) / (max_dist - beam_radius * 5))
                    target_size = min_cell_size * 2 + t * (max_cell_size - min_cell_size * 2)
                
                # Mark for refinement if cell is too large
                if cell_size > target_size * 1.5:
                    cell_markers[cell] = True
                    cells_marked += 1
            
            if cells_marked > 0:
                mesh = df.refine(mesh, cell_markers)
                print(f"  Graded refinement level {level+1}: refined {cells_marked} cells")
            else:
                break
        
        print(f"  Final graded mesh: {mesh.num_cells()} cells")
        return mesh

    def transfer_solution_with_projection(self, u_old, V_new):
        """Use projection instead of interpolation"""
        u_new = df.Function(V_new)
        
        # Project preserves weak form better than interpolation
        u_new = df.project(u_old, V_new)
        
        return u_new

    def _calculate_initial_resolution(self):
        """Calculate coarse initial mesh for AMR"""
        beam_radius = self.parameters['beam_radius']
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        # Domain to beam size ratio
        domain_size = min(L, W)
        size_ratio = domain_size / (2 * beam_radius)
        
        print(f"\n{'='*60}")
        print(f"ADAPTIVE MESH SETUP")
        print(f"{'='*60}")
        print(f"Domain: {L*1000:.1f} x {W*1000:.1f} x {H*1000:.2f} mm")
        print(f"Beam diameter: {2*beam_radius*1000:.3f} mm")
        print(f"Domain/beam ratio: {size_ratio:.1f}x")
        
        # Start with coarse mesh for AMR
        if size_ratio > 100:
            nx = ny = 20  # Very coarse for large domains
        elif size_ratio > 50:
            nx = ny = 25
        elif size_ratio > 20:
            nx = ny = 30
        else:
            nx = ny = 40  # Finer for smaller domains
        
        # Z resolution
        nz = max(8, int(15 * H / L))
        
        print(f"Initial coarse mesh: {nx} x {ny} x {nz}")
        print(f"Will refine adaptively during simulation")
        print(f"{'='*60}\n")
        
        return nx, ny, nz
    
    def _apply_initial_refinement(self):
        """Apply minimal initial refinement near laser"""
        laser_x = self.parameters.get('laser_x_position', self.parameters['length']/2)
        laser_y = self.parameters.get('laser_y_position', self.parameters['width']/2)
        beam_radius = self.parameters['beam_radius']
        
        # Only 1-2 initial refinements to keep mesh small
        max_initial_levels = 2
        initial_cells = self.mesh.num_cells()
        
        for level in range(max_initial_levels):
            if self.mesh.num_cells() > 50000:  # Keep initial mesh small
                break
            
            cell_markers = df.MeshFunction("bool", self.mesh, self.mesh.topology().dim())
            cell_markers.set_all(False)
            
            cells_marked = 0
            for cell in df.cells(self.mesh):
                midpoint = cell.midpoint()
                dist = np.sqrt((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)
                
                # Only refine very close to laser
                if dist < beam_radius * 2:
                    cell_markers[cell] = True
                    cells_marked += 1
            
            if cells_marked > 0:
                self.mesh = df.refine(self.mesh, cell_markers)
                print(f"Initial refinement {level+1}: {cells_marked} cells refined")
        
        print(f"Initial mesh: {initial_cells} → {self.mesh.num_cells()} cells")
    
    def adapt_mesh(self, temperature, time, max_cells=150000):
        """Adapt mesh based on temperature field"""
        mesh = temperature.function_space().mesh()
        
        # Get laser position (could be moving)
        laser_x = self.parameters.get('laser_x_position', self.parameters['length']/2)
        laser_y = self.parameters.get('laser_y_position', self.parameters['width']/2)
        beam_radius = self.parameters['beam_radius']
        
        # Check if adaptation is needed
        if not self._should_adapt(mesh, time):
            return mesh, False
        
        print(f"\nAdapting mesh at t={time:.3f}s (current: {mesh.num_cells()} cells)")
        
        # Calculate refinement indicators
        indicators = self._calculate_refinement_indicators(temperature, laser_x, laser_y, beam_radius)
        
        # Mark cells for refinement
        cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        
        cells_marked = self._mark_cells_for_refinement(
            mesh, indicators, cell_markers, laser_x, laser_y, beam_radius, max_cells
        )
        
        # Refine if needed
        if cells_marked > 0 and mesh.num_cells() + cells_marked * 7 < max_cells:
            new_mesh = df.refine(mesh, cell_markers)
            print(f"  Refined {cells_marked} cells → {new_mesh.num_cells()} total")
            return new_mesh, True
        else:
            if cells_marked > 0:
                print(f"  Cell limit reached, skipping refinement")
            return mesh, False
    
    def _should_adapt(self, mesh, time):
        """Determine if mesh adaptation should occur"""
        # Don't adapt too early (let solution develop)
        if time < 0.05:
            return False
        
        # Don't adapt if mesh is too small
        if mesh.num_cells() < 100:
            return False
        
        # Check time since last adaptation (stored in parameters)
        last_adapt_time = self.parameters.get('last_adapt_time', 0)
        adapt_interval = self.parameters.get('adapt_interval', 0.1)  # seconds
        
        if time - last_adapt_time < adapt_interval:
            return False
        
        return True
    
    def _calculate_refinement_indicators(self, temperature, laser_x, laser_y, beam_radius):
        """Calculate indicators for refinement"""
        mesh = temperature.function_space().mesh()
        
        # Temperature gradient indicator
        DG0 = df.FunctionSpace(mesh, "DG", 0)
        grad_T = df.grad(temperature)
        gradient_magnitude = df.project(df.sqrt(df.inner(grad_T, grad_T)), DG0)
        
        # Temperature value indicator
        temp_indicator = df.project(temperature, DG0)
        
        # Combined indicators
        indicators = {
            'gradient': gradient_magnitude.vector().get_local(),
            'temperature': temp_indicator.vector().get_local(),
            'beam_distance': np.zeros(mesh.num_cells())
        }
        
        # Calculate distance from beam center for each cell
        for cell in df.cells(mesh):
            midpoint = cell.midpoint()
            dist = np.sqrt((midpoint.x() - laser_x)**2 + (midpoint.y() - laser_y)**2)
            indicators['beam_distance'][cell.index()] = dist
        
        return indicators
    
    def _mark_cells_for_refinement(self, mesh, indicators, cell_markers, 
                                   laser_x, laser_y, beam_radius, max_cells):
        """Mark cells for refinement based on indicators"""
        cells_marked = 0
        
        # Get indicator arrays
        gradient = indicators['gradient']
        temperature = indicators['temperature']
        beam_distance = indicators['beam_distance']
        
        # Calculate thresholds
        gradient_threshold = np.percentile(gradient, 80)  # Top 20%
        temp_threshold = 500  # K, adjust based on material
        
        # Check beam resolution
        beam_cells = np.sum(beam_distance < beam_radius)
        target_beam_cells = 200  # Approximate target
        beam_needs_refinement = beam_cells < target_beam_cells
        
        for cell in df.cells(mesh):
            idx = cell.index()
            
            # Priority 1: Ensure beam is well resolved
            if beam_distance[idx] < beam_radius * 1.5 and beam_needs_refinement:
                cell_size = self._estimate_cell_size(cell)
                target_size = beam_radius / 10  # 20 cells across diameter
                if cell_size > target_size * 1.2:
                    cell_markers[cell] = True
                    cells_marked += 1
                    continue
            
            # Priority 2: High gradient regions
            if gradient[idx] > gradient_threshold:
                cell_markers[cell] = True
                cells_marked += 1
                continue
            
            # Priority 3: High temperature regions
            if temperature[idx] > temp_threshold and beam_distance[idx] < beam_radius * 10:
                cell_size = self._estimate_cell_size(cell)
                if cell_size > beam_radius / 5:  # Don't over-refine far field
                    cell_markers[cell] = True
                    cells_marked += 1
        
        # Limit refinement to prevent memory issues
        if cells_marked * 8 > max_cells - mesh.num_cells():
            # Too many cells marked, prioritize beam region
            print(f"  Limiting refinement: {cells_marked} → ", end='')
            cell_markers.set_all(False)
            cells_marked = 0
            
            # Only refine beam region
            for cell in df.cells(mesh):
                if beam_distance[cell.index()] < beam_radius * 2:
                    cell_markers[cell] = True
                    cells_marked += 1
            
            print(f"{cells_marked} cells")
        
        return cells_marked
    
    def _estimate_cell_size(self, cell):
        """Estimate cell size (simplified)"""
        return cell.h()  # Use built-in method
    
    def get_function_space(self):
        """Get function space for current mesh"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
        return df.FunctionSpace(self.mesh, 'P', 1)
    
    def calculate_stable_time_step(self):
        """Calculate stable time step for explicit schemes"""
        if self.mesh is None:
            raise RuntimeError("Mesh not created yet")
        
        # Material properties
        k = self.parameters.get('thermal_conductivity', 45.0)
        rho = self.parameters.get('density', 7850.0)
        cp = self.parameters.get('specific_heat', 460.0)
        
        # Thermal diffusivity
        alpha = k / (rho * cp)
        
        # Minimum cell size
        h_min = self.mesh.hmin()
        
        # CFL condition for heat equation
        safety_factor = 0.25
        dt_stable = safety_factor * h_min**2 / (2 * alpha)
        
        return dt_stable
    
    def update_adaptation_time(self, time):
        """Update last adaptation time"""
        self.parameters['last_adapt_time'] = time
    
    def print_mesh_statistics(self):
        """Print current mesh statistics"""
        if self.mesh is None:
            return
        
        print(f"\nMesh Statistics:")
        print(f"  Cells: {self.mesh.num_cells():,}")
        print(f"  Vertices: {self.mesh.num_vertices():,}")
        print(f"  Min cell size: {self.mesh.hmin()*1000:.3f} mm")
        print(f"  Max cell size: {self.mesh.hmax()*1000:.3f} mm")
        
        # Memory estimate
        memory_mb = self.mesh.num_cells() * 0.001  # Rough estimate
        print(f"  Estimated memory: ~{memory_mb:.0f} MB")