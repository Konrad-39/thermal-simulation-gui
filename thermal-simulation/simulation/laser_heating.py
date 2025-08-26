"""
3D Laser heating simulation using FEniCS.
Includes stationary heat source, phase change, and advanced heat transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import SimulationBase
import time
from scipy.interpolate import interp1d
import pandas as pd  
from .Mesh import ThermalMesh
from .timestep import TimeSteppingSolver

try:
    import dolfin as df
    from dolfin import *
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    print("Warning: FEniCS not available. Using fallback implementation.")

class LaserHeatingSimulation(SimulationBase):
    """3D Laser heating simulation using FEniCS"""
    
    def __init__(self):
        super().__init__()

        # Initialize handlers as None - will be created after parameters are set
        self.mesh_handler = None
        self.time_solver = None
        
        self.default_parameters = {
            # Material properties (need to set these to SiC)
            'k': 120.0,           # Thermal conductivity W/m·K
            'rho': 3200.0,       # Density kg/m³
            'cp': 750.0,         # Specific heat J/kg·K
            'T_ambient': 298.0,  # Ambient temperature K
            'T_melt': 3103.0,    # Melting temperature K
            'T_vaporization': 3134.0,  # Vaporization temperature K
            
            # Geometry (3D)
            'length': 0.008,      # m (x-direction)
            'width': 0.008,       # m (y-direction)
            'height': 0.0005,     # m (z-direction)
            'mesh_resolution': 35,  # Elements per dimension
            
            # Laser parameters (stationary at center)
            'peak_laser_power': 300.0,    # W
            'beam_radius': 0.0005,    # m
            'absorptivity': 0.95,     # Absorption coefficient
            'power_profile': 'constant', #constant or 'from_file'
            'power_file': '',  #path to power file

            # Heat transfer
            'convection_coeff': 0.0,  # W/m²·K
            'emissivity': 0.8,         # Surface emissivity
            'stefan_boltzmann': 5.67e-8, # W/m²·K⁴
            
            # Simulation
            'total_time': 10.0,     # s
            'dt': 0.001,          # s
            'output_interval': 10,  # steps
            'time_scale_factor': 1.0, #Accelleration factor
            'scale_thermal_properties': True,
            'output_time_interval': 0.5  # Add this line

        }

        # Initialize power interpolator
        self.power_interpolator = None

        self.parameters = self.default_parameters.copy()
        
        self.parameters.update({
            'adaptive_refinement': True,
            'refinement_interval': 10,  # Refine every N steps
            'refinement_threshold': 0.6,  # Fraction of max gradient for refinement
            'max_refinement_levels': 3,  # Prevent infinite refinement
            'min_cell_size': 1e-5,  # Minimum allowed cell size
        })

        if 'beam_radius' in self.parameters:
            self.parameters['min_cell_size'] = self.parameters['beam_radius'] / 50


        # FEniCS objects
        self.mesh = None
        self.V = None
        self.u = None
        self.u_n = None
        self.laser_source = None

        # Initialize handlers after parameters are set
        self.initialize_handlers()

    def initialize_handlers(self):
        """Initialize mesh and time stepping handlers after parameters are set"""
        try:
            if self.mesh_handler is None:
                self.mesh_handler = ThermalMesh(self.parameters)
                print("Mesh handler initialized")
            
            # Don't initialize time_solver here yet - wait until we have a mesh
            print("Handlers initialization completed")
            
        except Exception as e:
            print(f"Error initializing handlers: {e}")
            # Set to None to prevent further errors
            self.mesh_handler = None
            self.time_solver = None

    def load_power_profile(self):
        """Load power profile from CSV or Excel file"""
        if self.parameters['power_profile'] != 'from_file':
            print("DEBUG: Power profile is not 'from_file', skipping file load")
            return
            
        power_file = self.parameters['power_file']
        
        if not power_file:
            raise ValueError("Power file path is required when using 'from_file' power profile")
        
        try:
            # Try to read as CSV first
            if power_file.lower().endswith('.csv'):
                df = pd.read_csv(power_file)
            elif power_file.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(power_file)
            else:
                # Try CSV as default
                df = pd.read_csv(power_file)
            
            # Expect first column to be time, second to be power percentage
            if df.shape[1] < 2:
                raise ValueError("Power file must have at least 2 columns: time and power percentage")
            
            times = df.iloc[:, 0].values
            power_percentages = df.iloc[:, 1].values
            
            # Validate data
            if len(times) == 0:
                raise ValueError("Power file contains no data")
            
            if np.any(power_percentages < 0) or np.any(power_percentages > 100):
                raise ValueError("Power percentages must be between 0 and 100")
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            power_percentages = power_percentages[sort_idx]
            
            power_derivatives = np.gradient(power_percentages, times)
            peak_power = self.parameters['peak_laser_power']
            power_derivatives_watts = power_derivatives*peak_power/100

            # Create interpolator
            scaled_times = self.get_scaled_time(times)

            self.power_times = times
            self.power_interpolator = interp1d(
                scaled_times, power_percentages, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(power_percentages[0], power_percentages[-1])
            )
            
            print(f"Loaded power profile with {len(times)} points")
            print(f"Time range: {times[0]:.3f} - {times[-1]:.3f} s")
            print(f"Power range: {power_percentages.min():.1f} - {power_percentages.max():.1f} %")
            
            self.power_derivative_interpolator = interp1d(
                times, np.abs(power_derivatives_watts),
                kind='linear',
                bounds_error=False,
                fill_value=(0, 0)
            )
            
            print(f"Max power change rate: {np.max(np.abs(power_derivatives)):.1f} W/s")

        except Exception as e:
            raise ValueError(f"Error loading power profile file: {str(e)}")
    
    def get_laser_power(self, t):
        """Get laser power at time t"""
        peak_power = self.parameters['peak_laser_power']
        
        if self.parameters['power_profile'] == 'constant':
            # Use 100% of peak power for constant profile
            return peak_power
        elif self.parameters['power_profile'] == 'from_file':
            if self.power_interpolator is None:
                print(f"Warning: Power profile set to 'from_file' but no file loaded. Using constant power: {peak_power}W")
                return peak_power
            # Get power percentage and convert to actual power
            power_percent = self.power_interpolator(t)
            return peak_power * (power_percent / 100.0)
        else:
            raise ValueError(f"Unknown power profile: {self.parameters['power_profile']}") 
    
    def get_power_change_rate(self, t):
        """Get power change rate at time t"""
        if hasattr(self, 'power_derivative_interpolator'):
            return self.power_derivative_interpolator(t)
        else:
            return 0.0

    def apply_time_scaling(self):
        """Apply time scaling with actual computational speedup"""
        if not self.parameters.get('scale_thermal_properties', False):
            return
        
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        if scale_factor == 1.0:
            return
        
        # Store originals
        self.parameters['original_total_time'] = self.parameters['total_time']
        self.parameters['original_dt'] = self.parameters['dt']
        self.parameters['original_rho'] = self.parameters['rho']
        self.parameters['original_cp'] = self.parameters['cp']
        
        print(f"=== SMART TIME SCALING: {scale_factor}x ===")
        
        # Calculate original thermal diffusivity
        k = self.parameters['k']
        alpha_original = k / (self.parameters['rho'] * self.parameters['cp'])
        print(f"Original α: {alpha_original:.2e} m²/s")
        
        # Scale thermal properties to make heat diffuse faster
        sqrt_factor = scale_factor ** 0.5
        self.parameters['rho'] = self.parameters['rho'] / sqrt_factor
        self.parameters['cp'] = self.parameters['cp'] / sqrt_factor
        
        # Calculate new thermal diffusivity
        alpha_new = k / (self.parameters['rho'] * self.parameters['cp'])
        print(f"New α: {alpha_new:.2e} m²/s (increased {alpha_new/alpha_original:.0f}x)")
        
        print(f"Simulating {self.parameters['total_time']}s with {scale_factor}x faster physics")
        print("=" * 50) 
   
    def get_scaled_time(self, simulation_time):
        """Convert simulation time back to real time for output"""
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        return simulation_time * scale_factor

    def get_real_time(self, scaled_time):
        """Convert computational time to real time"""
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        if scale_factor == 1.0:
            return scaled_time
        
        # Real time should be scaled UP from computational time
        return scaled_time * scale_factor

    def should_output_at_time(self, current_time, existing_times):
        """Determine if we should output results at current time based on time interval"""
        output_interval = self.parameters.get('output_time_interval', 0.5)
        
        # Always output at t=0
        if len(existing_times) == 0:
            return True
        
        # Check if enough time has passed since last output
        last_output_time = existing_times[-1] if existing_times else 0
        return (current_time - last_output_time) >= output_interval

    def setup_fenics_problem(self):
        """Setup 3D FEniCS mesh and function spaces"""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS is required for this simulation")
            
        comm = None
        
        try:    
            self.load_power_profile()
        
            # Ensure handlers are initialized
            if self.mesh_handler is None:
                print("Initializing handlers in setup_fenics_problem...")
                self.initialize_handlers()
            
            if self.mesh_handler is None:
                raise RuntimeError("Failed to initialize mesh handler")
            
            # Create mesh using mesh handler
            self.mesh = self.mesh_handler.create_mesh(None)
            
            # NOW initialize time solver with the mesh
            if self.time_solver is None:
                self.time_solver = TimeSteppingSolver(self.parameters, None, self.mesh_handler)
                print("Time solver initialized")
            
            # Check stability
            self.time_solver.check_stability()

            L = self.parameters['length']
            W = self.parameters['width'] 
            H = self.parameters['height']
            
            # Material properties
            k = self.parameters['k']
            rho = self.parameters['rho']
            cp = self.parameters['cp']
            dt = self.parameters['dt']
            h_conv = self.parameters['convection_coeff']
            T_ambient = self.parameters['T_ambient']

            volume = L * W * H
            mass = rho * volume
            power = self.parameters['peak_laser_power']
            
            # Estimate temperature rise per second
            estimated_temp_rise_per_sec = power / (mass * cp)
            print(f"Expected temperature rise: {estimated_temp_rise_per_sec:.1f} K/s")
            print(f"In {dt} seconds: {estimated_temp_rise_per_sec * dt:.3f} K")
           
            # Define function space
            self.V = df.FunctionSpace(self.mesh, 'P', 1)
            
            # Define Test function
            v = df.TestFunction(self.V)

            # Solution functions
            self.u = df.Function(self.V) # current solution
            self.u_n = df.Function(self.V) # previous time step solution

            # Set initial conditions
            self.u.assign(df.Constant(T_ambient))
            self.u_n.assign(df.Constant(T_ambient))

            # Laser heat source (updated every time step)
            self.laser_source = df.Function(self.V)

            # No Dirichlet BC - using natural BCs
            self.bcs = []

            # Main function (Time dependent conduction)
            self.dt_param = df.Constant(self.parameters['dt'])
            self.time_solver.set_dt_param(self.dt_param)


            self.F = (rho*cp*(self.u-self.u_n)*v + self.dt_param*k*df.dot(df.grad(self.u),df.grad(v)))*df.dx

            # Radiation
            emissivity = self.parameters['emissivity']
            sigma = self.parameters['stefan_boltzmann']
            
            self.F += self.dt_param * emissivity * sigma * (self.u**4 - T_ambient**4) * v * df.ds

            # Add convective boundary term
            if h_conv > 0:
                self.F += self.dt_param*h_conv*(self.u-T_ambient)*v*df.ds

            # Add heat source term
            self.F -= self.dt_param*self.laser_source*v*df.dx

            self.solver_params = {
                'nonlinear_solver': 'newton',
                'newton_solver': {
                    'relative_tolerance': 1e-6,
                    'absolute_tolerance': 1e-10,
                    'maximum_iterations': 100,
                    'linear_solver': 'mumps',
                    'preconditioner': 'default',
                    'relaxation_parameter': 0.8,
                    'error_on_nonconvergence': False
                }
            }
    
        except Exception as e:
            raise RuntimeError(f"Failed to set up simulation: {str(e)}")


    def recreate_variational_form_after_refinement(self):
        """Recreate variational form after mesh refinement"""
        # Material properties
        k = self.parameters['k']
        rho = self.parameters['rho']
        cp = self.parameters['cp']
        h_conv = self.parameters['convection_coeff']
        T_ambient = self.parameters['T_ambient']
        emissivity = self.parameters['emissivity']
        sigma = self.parameters['stefan_boltzmann']
        
        # Test function
        v = df.TestFunction(self.V)
        
        # Recreate the variational form with updated function space
        self.F = (rho*cp*(self.u-self.u_n)*v + self.dt_param*k*df.dot(df.grad(self.u),df.grad(v)))*df.dx
        self.F += self.dt_param * emissivity * sigma * (self.u**4 - T_ambient**4) * v * df.ds

        if h_conv > 0:
            self.F += self.dt_param*h_conv*(self.u-T_ambient)*v*df.ds
        
        self.F -= self.dt_param*self.laser_source*v*df.dx
        
    def update_laser_source(self, t_real):
        """Update laser heat source for current time step"""
        if self.power_interpolator is not None:
            current_power = self.power_interpolator(t_real)
        else:
            current_power = self.parameters['peak_laser_power']

        # Laser is stationary at center of mesh
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        laser_x = L / 2  # Center of mesh
        laser_y = W / 2  # Center of mesh
        laser_z = H      # Top surface
        
        # Laser parameters
        rt = self.get_scaled_time(t_real)
        if self.power_interpolator is not None:
            power = self.get_laser_power(rt)
        else:
            power = self.parameters['peak_laser_power']

        radius = self.parameters['beam_radius']
        absorptivity = self.parameters['absorptivity']
        
        # Define Gaussian heat source
        class LaserSource(df.UserExpression):
            def __init__(self, laser_x, laser_y, laser_z, power, radius, absorptivity, **kwargs):
                super().__init__(**kwargs)
                self.laser_x = laser_x
                self.laser_y = laser_y
                self.laser_z = laser_z
                self.power = power
                self.radius = radius
                self.absorptivity = absorptivity
                
            def eval(self, value, x):
                # Distance from laser center
                r_squared = (x[0] - self.laser_x)**2 + (x[1] - self.laser_y)**2
            
                # Gaussian distribution in x-y plane
                intensity = (2* self.power * self.absorptivity) / (np.pi * self.radius**2)
                gaussian = np.exp(-2 * r_squared / self.radius**2)
                
                # Depth attenuation (Beer-Lambert law)
                penetration_depth = 1e-5  # m 
                depth_from_surface = self.laser_z - x[2]
        
                if depth_from_surface >= 0 and depth_from_surface <= 5 * penetration_depth:
                    depth_factor = np.exp(-depth_from_surface / penetration_depth)
                    volumetric_intensity = intensity * gaussian * depth_factor / penetration_depth
                    value[0] = volumetric_intensity
                else:
                    value[0] = 0.0   

            def value_shape(self):
                return ()
                
        # Create and interpolate laser source
        laser_expr = LaserSource(laser_x, laser_y, laser_z, power, radius, absorptivity, degree=2)
        self.laser_source.interpolate(laser_expr)

        print(f"Laser at ({laser_x*1000:.2f}, {laser_y*1000:.2f}, {laser_z*1000:.2f}) mm")
        print(f"Mesh bounds: (0, 0, 0) to ({L*1000:.2f}, {W*1000:.2f}, {H*1000:.2f}) mm")
        print(f"Power: {power:.1f}W, Radius: {radius*1000:.2f}mm, Absorptivity: {absorptivity}")
        
    def get_laser_position(self, t):
        """Get laser position at time t (stationary at center) Could be changed to move the laser"""
        L = self.parameters['length']
        W = self.parameters['width']
        return L / 2, W / 2  # Always at center

    def run(self, progress_callback=None, use_parallel=True):
        """Run the 3D laser heating simulation with optional parallel processing"""
        if not FENICS_AVAILABLE:
            return self._run_fallback(progress_callback)
        
        
        rank = 0
    
 # No MPI available, use defaults

        # Validate parameters
        errors = self.validate_parameters()
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        
        try:
            
            print("Starting 3D FEniCS laser heating simulation...")

            # Apply time scaling before starting
            self.apply_time_scaling()

            # Setup FEniCS problem
            self.setup_fenics_problem()

            # Get time stepping info
            time_info = self.time_solver.get_time_stepping_info()
            
            # Get scaling parameters
            scale_factor = self.parameters.get('time_scale_factor', 1.0)
            
            # Time stepping parameters
            total_time = time_info['total_time']
            base_dt = time_info['dt']
            output_interval = time_info['output_interval']

            # Storage for results
            times = []
            max_temperatures = []
            temperature_fields = []
            
            # Time stepping loop
            t_scaled = 0
            step = 0
            prev_max_temp = self.parameters['T_ambient']

            while self.get_real_time(t_scaled) < total_time:
                if self.stop_requested:
                    break
                    
                # Get current temperature
                self.current_time = t_scaled 
                current_max_temp = self.u.vector().max() if hasattr(self, 'u') else prev_max_temp
                
                # Calculate adaptive time step using time solver
                dt_scaled = self.time_solver.calculate_optimal_timestep(
                    t_scaled, current_max_temp, prev_max_temp, base_dt, self.power_interpolator
                )
                
                # Don't overshoot total time
                if t_scaled + dt_scaled > total_time:
                    dt_scaled = total_time - t_scaled
                
                t_scaled += dt_scaled
                step += 1

                # Update time step using time solver
                self.time_solver.update_dt(dt_scaled)
                t_real = self.get_real_time(t_scaled)
                self.update_laser_source(t_scaled)
                
                # Solve time step using time solver
                success, error = self.time_solver.solve_time_step(self.F, self.u, self.bcs, step, t_scaled)
                
                if not success:
                    if rank == 0:
                        print(error)
                    break

                current_max_temp = self.u.vector().max()
                prev_max_temp = current_max_temp

                # Check for mesh refinement using mesh handler
                if self.time_solver.should_refine_mesh(step):
                    try:
                        # Get laser criteria for refinement
                        laser_criteria = {
                            'laser_position': self.get_laser_position(t_scaled),
                            'laser_radius': self.parameters['beam_radius'],
                            'T_melt': self.parameters['T_melt'],
                            'current_time': t_scaled
                        }
                        
                        mesh_changed = self.mesh_handler.adaptive_mesh_refinement(self.u, laser_criteria)
                        if mesh_changed:
                            print(f"Mesh refined at step {step}, time {t_scaled:.4f}s")
                            
                            print("DEBUG: About to update function space...")
                            # Update function space and recreate variational form
                            self.V = self.mesh_handler.get_function_space()
                            print("DEBUG: Function space updated")
                            
                            print("DEBUG: About to update mesh reference...")
                            self.mesh = self.mesh_handler.mesh  # Update mesh reference
                            print("DEBUG: Mesh reference updated")
                            
                            print("DEBUG: About to recreate variational form...")
                            self.recreate_variational_form_after_refinement()
                            print("DEBUG: Variational form recreated")
                            
                            print("DEBUG: About to update laser source...")
                            self.update_laser_source(t_scaled)
                            print("DEBUG: Laser source updated")
                            
                    except Exception as e:
                        print(f"Post-refinement error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue without the updates
                    if step % 5 == 0 or step < 20:  # Report frequently at start, then every 5 steps
                        if rank == 0:
                            print(f"Step {step:4d}, Real Time: {t_real:.4f}s, "
                                f"Scaled Time: {t_scaled:.6f}s, Max Temp: {current_max_temp:.1f}K")

                # Store results
                # if self.time_solver.should_output_results(step,t_real):
                #     times.append(t_real)  # Store real time
                #     max_temperatures.append(current_max_temp)
                #     temp_values = self.u.vector().get_local()
                #     temperature_fields.append(temp_values.tolist())

                #     if progress_callback and rank == 0:
                #         real_progress = (t_real / self.parameters['total_time']) * 100
                #         progress_callback(real_progress)
                #         print(f"Progress: {real_progress:.1f}% (t_real={t_real:.4f}s)")

                if self.should_output_at_time(t_real, times):
                    times.append(t_real)  # Store real time
                    max_temperatures.append(current_max_temp)
                    temp_values = self.u.vector().get_local()
                    temperature_fields.append(temp_values.tolist())

                    if progress_callback:
                        real_progress = (t_real / self.parameters['total_time']) * 100
                        progress_callback(real_progress)
                        print(f"Progress: {real_progress:.1f}% (t_real={t_real:.4f}s)")

                # Update for next step
                self.u_n.assign(self.u)
                prev_max_temp = current_max_temp
                
            print("Skipping post-processing for faster results...")
            melt_volumes = [0.0] * len(times)
            heat_fluxes = [0.0] * len(times)
            
            # Compile results (only on rank 0 for MPI)
            if rank == 0 or comm is None:
                results = {
                    'times': times,
                    'max_temperatures': max_temperatures,
                    'melt_pool_volumes': melt_volumes,
                    'heat_fluxes': heat_fluxes,
                    'temperature_fields': temperature_fields,
                    'final_temperature_function': self.u.copy(deepcopy=True),
                    'parameters': self.parameters.copy()
                }
                return results
            else:
                return None
                
        except Exception as e:
            if rank == 0:
                print(f"Simulation error: {e}")
            raise

    def _run_fallback(self, progress_callback=None):
        """Fallback implementation when FEniCS is not available"""
        print("Warning: Running fallback simulation (FEniCS not available)")
        
        # Simple analytical approximation
        total_time = self.parameters['total_time']
        dt = self.parameters['dt']
        n_steps = int(total_time / dt)
        
        times = []
        max_temperatures = []
        
        T_ambient = self.parameters['T_ambient']
        peak_power = self.parameters['peak_laser_power']
        
        for step in range(n_steps):
            t = step * dt
            times.append(t)
            
            # Simple exponential heating model
            temp_rise = (peak_power / 1000) * (1 - np.exp(-t/0.1))  # Simplified
            max_temp = T_ambient + temp_rise
            max_temperatures.append(max_temp)
            
            if progress_callback:
                progress_callback((step + 1) / n_steps * 100)
        
        return {
            'times': times,
            'max_temperatures': max_temperatures,
            'melt_pool_volumes': [0.0] * len(times),
            'heat_fluxes': [0.0] * len(times),
            'temperature_fields': [],
            'parameters': self.parameters.copy()
        }

    def extract_temperature_field(self):
        """Extract temperature field on a regular grid for visualization"""
        # Create regular grid for visualization
        nx = ny = 50
        nz = 20
        
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        x = np.linspace(0, L, nx)
        y = np.linspace(0, W, ny)
        z = np.linspace(0, H, nz)
        
        # Extract temperature at grid points
        temp_field = np.zeros((nx, ny, nz))
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    try:
                        point = df.Point(xi, yj, zk)
                        temp_field[i, j, k] = self.u(point)
                    except:
                        temp_field[i, j, k] = self.parameters['T_ambient']
                        
        return {
            'x': x,
            'y': y,
            'z': z,
            'temperature': temp_field
        }
        
    def plot_results(self, results, axes):
            """
            Plots for 3D FEniCS simulation. Heat maps of surface and 
            cross section thermal gradients, average temperature in the laser spot size
            """
            ax1, ax2, ax3, ax4 = axes
            
            # Get geometry parameters
            L = self.parameters['length']
            W = self.parameters['width']
            H = self.parameters['height']
            
            # Use the stored final temperature function for spatial plots
            final_temp_func = results.get('final_temperature_function', None)
            
            # Plot 1: Surface temperature heat map (laser side - top surface)
            ax1.clear()
            if hasattr(self, 'u') and self.u is not None:
                surface_data = self.extract_surface_temperature_from_function(final_temp_func)

                if surface_data:
                    X, Y = surface_data['X'], surface_data['Y']
                    surface_temp = surface_data['temperature']
                    
                    # Create color map
                    im = ax1.contourf(X*1000, Y*1000, surface_temp, levels=50, cmap='hot')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax1, label='Temperature (K)', shrink=0.8)
                    
                    # Add melting contour
                    T_melt = self.parameters['T_melt']
                    if np.max(surface_temp) > T_melt:
                        ax1.contour(X*1000, Y*1000, surface_temp, levels=[T_melt], 
                                colors='cyan', linewidths=2, linestyles='--')
                    
                    ax1.set_xlabel('X (mm)')
                    ax1.set_ylabel('Y (mm)')
                    ax1.set_title('Surface Temperature at Final Time')
                    ax1.set_aspect('equal')
                else:
                    ax1.set_title('Surface Temperature - No Data Available')
            
            # Plot 2: Cross-section through thickness (X-Z plane at centerline)
            ax2.clear()
            if final_temp_func is not None:
                self.plot_xz_cross_section_from_function(ax2, final_temp_func)
            else:
                ax2.set_title('X-Z Cross-Section - No Data Available')

            # Plot 3: Cross-section through thickness (Y-Z plane at centerline)
            ax3.clear()
            if final_temp_func is not None:
                self.plot_yz_cross_section_from_function(ax3, final_temp_func)
            else:
                ax3.set_title('Y-Z Cross-Section - No Data Available')

            # Plot 4: Temperature vs time at laser spot
            ax4.clear()
            if 'times' in results and 'max_temperatures' in results:
                ax4.plot(results['times'], results['max_temperatures'], 'r-', linewidth=2, label='Max Temperature')
                
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Temperature (K)')
                ax4.set_title('Maximum Temperature vs Time')
                ax4.grid(True, alpha=0.3)
                ax4.legend()

    def extract_surface_temperature_from_function(self, temp_function):
        """Extract temperature on the top surface using provided function"""
        if temp_function is None:
            return None
        
        # Get geometry parameters
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        # Create surface grid
        nx = ny = 50
        x = np.linspace(0, L, nx)
        y = np.linspace(0, W, ny)
        X, Y = np.meshgrid(x, y)
        
        # Extract temperature at each surface point
        surface_temp = np.zeros((ny, nx))
        
        for i in range(ny):
            for j in range(nx):
                try:
                    # Point on top surface (z = H)
                    point = df.Point(X[i, j], Y[i, j], H)
                    surface_temp[i, j] = temp_function(point)
                except RuntimeError:
                    # Point outside mesh domain
                    surface_temp[i, j] = self.parameters['T_ambient']
        
        return {
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'temperature': surface_temp
        }

    def plot_xz_cross_section_from_function(self, ax, temp_function):
        """Plot temperature cross-section in X-Z plane using provided function"""
        if temp_function is None:
            return
        
        # Get geometry parameters
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        # Cross-section at centerline (y = W/2)
        y_cross = W / 2
        
        # Create grid
        nx, nz = 80, 40
        x = np.linspace(0, L, nx)
        z = np.linspace(0, H, nz)
        X, Z = np.meshgrid(x, z)
        
        # Extract temperature
        temp_xz = np.zeros((nz, nx))
        
        for i in range(nz):
            for j in range(nx):
                try:
                    point = df.Point(X[i, j], y_cross, Z[i, j])
                    temp_xz[i, j] = temp_function(point)
                except RuntimeError:
                    temp_xz[i, j] = self.parameters['T_ambient']
        
        # Create heat map
        im = ax.contourf(X*1000, Z*1000, temp_xz, levels=50, cmap='hot')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Temperature (K)', shrink=0.8)
        
        # Add melting temperature contour
        T_melt = self.parameters['T_melt']
        if np.max(temp_xz) > T_melt:
            ax.contour(X*1000, Z*1000, temp_xz, levels=[T_melt], 
                    colors='cyan', linewidths=2, linestyles='--')
        
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Depth (mm)')
        ax.set_title(f'X-Z Cross-Section at Y={y_cross*1000:.1f} mm')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def plot_yz_cross_section_from_function(self, ax, temp_function):
        """Plot temperature cross-section in Y-Z plane using provided function"""
        if temp_function is None:
            return
        
        # Get geometry parameters
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        # Cross-section at centerline (x = L/2)
        x_cross = L / 2
        
        # Create grid
        ny, nz = 80, 40
        y = np.linspace(0, W, ny)
        z = np.linspace(0, H, nz)
        Y, Z = np.meshgrid(y, z)
        
        # Extract temperature
        temp_yz = np.zeros((nz, ny))
        
        for i in range(nz):
            for j in range(ny):
                try:
                    point = df.Point(x_cross, Y[i, j], Z[i, j])
                    temp_yz[i, j] = temp_function(point)
                except RuntimeError:
                    temp_yz[i, j] = self.parameters['T_ambient']
        
        # Create heat map
        im = ax.contourf(Y*1000, Z*1000, temp_yz, levels=50, cmap='hot')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Temperature (K)', shrink=0.8)
        
        # Add melting temperature contour
        T_melt = self.parameters['T_melt']
        if np.max(temp_yz) > T_melt:
            ax.contour(Y*1000, Z*1000, temp_yz, levels=[T_melt], 
                    colors='cyan', linewidths=2, linestyles='--')
        
        ax.set_xlabel('Y Position (mm)')
        ax.set_ylabel('Depth (mm)')
        ax.set_title(f'Y-Z Cross-Section at X={x_cross*1000:.1f} mm')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def validate_parameters(self):
        """Validate parameters specific to laser heating simulation"""
        errors = []
        print(f"DEBUG: Validating parameters: {self.parameters}")
    
        # Check if peak_laser_power exists and is positive
        if 'peak_laser_power' not in self.parameters:
            errors.append("Missing parameter: peak_laser_power")
        elif self.parameters['peak_laser_power'] <= 0:
            errors.append(f"peak_laser_power must be positive, got: {self.parameters['peak_laser_power']}")
       
        # Check required parameters
        required = ['k', 'rho', 'cp', 'T_ambient', 'peak_laser_power', 'beam_radius',
                   'absorptivity', 'length', 'width', 'height', 'total_time', 'dt']
        
        for param in required:
            if param not in self.parameters:
                errors.append(f"Missing parameter: {param}")
                
        # Check positive values
        positive_params = ['k', 'rho', 'cp', 'peak_laser_power', 'beam_radius',
                          'length', 'width', 'height', 'total_time', 'dt']
        
        for param in positive_params:
            if param in self.parameters and self.parameters[param] <= 0:
                errors.append(f"Parameter {param} must be positive")
                
        # Check ranges
        if 'absorptivity' in self.parameters:
            if not (0 <= self.parameters['absorptivity'] <= 1):
                errors.append("Absorptivity must be between 0 and 1")
                
        return errors

    def cleanup(self):
        """Clean up resources to free memory"""
        # Call parent cleanup
        super().cleanup()
        
        # Clean up laser-specific attributes
        if hasattr(self, 'laser_source'):
            del self.laser_source
        if hasattr(self, 'F'):
            del self.F
        if hasattr(self, 'bcs'):
            del self.bcs

