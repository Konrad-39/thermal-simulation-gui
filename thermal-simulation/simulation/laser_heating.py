# laser_heating.py
"""
3D Laser heating simulation using FEniCS.
Includes stationary heat source, phase change, and advanced heat transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import SimulationBase
from scipy.interpolate import interp1d
import pandas as pd  
from .Mesh import ThermalMesh
from .timestep import TimeSteppingSolver
from .equation_parser import EquationParser  # NEW IMPORT

# Remove these imports (now in equation_parser.py):
# import ast
# import operator  
# import math

try:
    import dolfin as df
    from dolfin import *
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    print("Warning: FEniCS not available. Using fallback implementation.")

class LaserHeatingSimulation(SimulationBase):
    """
    3D Laser heating simulation using FEniCS with power scaling for computational efficiency.
    
    Features:
    - Gaussian laser heat source with Beer-Lambert absorption
    - Power scaling for accelerated thermal diffusion
    - Temperature-dependent material properties (optional)
    - Surface temperature analysis and pyrometer simulation
    - Real-time adaptive time stepping
    
    Args:
        None - uses default parameters, call set_parameters() to customize
        
    Example:
        sim = LaserHeatingSimulation()
        sim.set_parameters({'peak_laser_power': 500.0, 'time_scale_factor': 1000.0})
        results = sim.run()
        sim.plot_results(results, axes)
    """
    
    def __init__(self):
        super().__init__()

        # Initialize handlers as None - will be created after parameters are set
        self.mesh_handler = None
        self.time_solver = None
        self.colorbars = []
        self.default_parameters = {
            # Material properties (SiC)
            'k': 120.0,           # Thermal conductivity W/m·K
            'rho': 3200.0,        # Density kg/m³
            'cp': 750.0,          # Specific heat J/kg·K
            'T_ambient': 298.0,   # Ambient temperature K
            'T_melt': 3103.0,     # Melting temperature K
            
            # Temperature-dependent properties (advanced users)
            'use_temperature_dependent_properties': False,
            'property_variation_type': 'custom_equation',
            'k_equation': '1600/T**(0.85) + 400/T',
            'rho_equation': 'rho_base*(1-(1.2*10**-5)*(T-298))',
            'cp_equation': '316.6 + 2.44*T-(1.65*10**5)/T**2',
            'k_base': 120.0,
            'rho_base': 3200.0,
            'cp_base': 750.0,
            
            # Geometry (3D)
            'length': 0.008,      # m (x-direction)
            'width': 0.008,       # m (y-direction)
            'height': 0.0005,     # m (z-direction)
            'mesh_resolution': 35,
            
            # Laser parameters
            'peak_laser_power': 300.0,    # W
            'beam_radius': 0.0005,        # m
            'absorptivity': 0.95,         # Absorption coefficient
            'power_profile': 'constant',  # 'constant' or 'from_file'
            'power_file': '',             # Path to power file

            # Heat transfer
            'convection_coeff': 0.0,      # W/m²·K
            'emissivity': 0.8,            # Surface emissivity
            'stefan_boltzmann': 5.67e-8,  # W/m²·K⁴
            
            # Simulation
            'total_time': 10.0,           # s
            'dt': 0.001,                  # s
            'output_interval': 10,        # steps
            'output_time_interval': 0.5,  # s
            'time_scale_factor': 1.0,     # Acceleration factor
            'scale_thermal_properties': True,
        }

        # Initialize power interpolator
        self.power_interpolator = None

        self.parameters = self.default_parameters.copy()
    

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
                df = pd.read_csv(power_file, header = None)
            elif power_file.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(power_file, header=None)
            else:
                # Try CSV as default
                df = pd.read_csv(power_file,header=None)
            
            # Expect first column to be time, second to be power percentage
            if df.shape[1] < 2:
                raise ValueError("Power file must have at least 2 columns: time and power percentage")
            
            times = df.iloc[:, 0].values
            power_percentages = df.iloc[:, 1].values
            

            times = pd.to_numeric(times, errors='coerce')
            power_percentages = pd.to_numeric(power_percentages, errors='coerce')
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
                times, power_percentages,  #scaled_times old value
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
        
        # Store originals BEFORE scaling
        self.parameters['original_total_time'] = self.parameters['total_time']
        self.parameters['original_dt'] = self.parameters['dt']
        self.parameters['original_k'] = self.parameters['k']      # IMPORTANT!
        self.parameters['original_rho'] = self.parameters['rho']
        self.parameters['original_cp'] = self.parameters['cp']
        
        print(f"=== SMART TIME SCALING: {scale_factor}x ===")
        print(f"Stored original values: k={self.parameters['original_k']}, "
            f"rho={self.parameters['original_rho']}, cp={self.parameters['original_cp']}")
        
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
        output_interval = self.parameters.get('output_time_interval', 0.1)  # Reduced to 0.1s
        max_outputs = 1000  # Limit total outputs
        
        # Always output at t=0
        if len(existing_times) == 0:
            return True
        
        # Don't exceed max outputs
        if len(existing_times) >= max_outputs:
            return False
        
        # Check if enough time has passed since last output
        last_output_time = existing_times[-1] if existing_times else 0
        time_since_last = current_time - last_output_time
        
        # Output if interval passed OR if this is near the end of simulation
        total_time = self.parameters.get('total_time', 10.0)
        near_end = current_time > (total_time * 0.9)  # Last 10% of simulation
        
        should_output = (time_since_last >= output_interval) or near_end
        
        return should_output

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
            # self.mesh = self.mesh_handler.create_mesh(None)
            self.mesh = self.mesh_handler.create_statisic_optimized_mesh(None)

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

            # Radiation
            emissivity = self.parameters['emissivity']
            sigma = self.parameters['stefan_boltzmann']

            # Check if using temperature-dependent properties
            if self.parameters.get('use_temperature_dependent_properties', False):
                print("Setting up temperature-dependent variational form...")
                self.setup_temperature_dependent_form(v, T_ambient, h_conv, emissivity, sigma)
            else:
                print("Using constant material properties...")
                # Use constant properties (your existing code)
                self.F = (rho*cp*(self.u-self.u_n)*v + self.dt_param*k*df.dot(df.grad(self.u),df.grad(v)))*df.dx
                self.F += self.dt_param * emissivity * sigma * (self.u**4 - T_ambient**4) * v * df.ds
                if h_conv > 0:
                    self.F += self.dt_param*h_conv*(self.u-T_ambient)*v*df.ds
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
        
    def setup_temperature_dependent_form(self, v, T_ambient, h_conv, emissivity, sigma):
        """Setup variational form with temperature-dependent properties"""
        # Create functions to hold the spatially-varying properties
        self.k_func = df.Function(self.V)
        self.rho_func = df.Function(self.V)
        self.cp_func = df.Function(self.V)
        
        # Initialize with base values
        k_base = self.parameters['k_base']
        rho_base = self.parameters['rho_base']
        cp_base = self.parameters['cp_base']
        
        self.k_func.assign(df.Constant(k_base))
        self.rho_func.assign(df.Constant(rho_base))
        self.cp_func.assign(df.Constant(cp_base))
        
        # Variational form with spatially-varying properties
        self.F = (self.rho_func * self.cp_func * (self.u - self.u_n) * v + 
                self.dt_param * self.k_func * df.dot(df.grad(self.u), df.grad(v))) * df.dx
        
        # Boundary terms remain the same
        self.F += self.dt_param * emissivity * sigma * (self.u**4 - T_ambient**4) * v * df.ds
        if h_conv > 0:
            self.F += self.dt_param * h_conv * (self.u - T_ambient) * v * df.ds
        self.F -= self.dt_param * self.laser_source * v * df.dx

    def update_material_properties(self):
        """Update material properties based on current temperature field"""
        if not self.parameters.get('use_temperature_dependent_properties', False):
            return
        
        # Get temperature values at all nodes
        temp_values = self.u.vector().get_local()
        
        # Calculate properties at each node
        k_values = np.zeros_like(temp_values)
        rho_values = np.zeros_like(temp_values)
        cp_values = np.zeros_like(temp_values)
        
        for i, T in enumerate(temp_values):
            props = self.get_material_properties(T)
            k_values[i] = props['k']
            rho_values[i] = props['rho']
            cp_values[i] = props['cp']
        
        # Update the property functions
        self.k_func.vector().set_local(k_values)
        self.rho_func.vector().set_local(rho_values)
        self.cp_func.vector().set_local(cp_values)
        
        # Apply changes
        self.k_func.vector().apply('insert')
        self.rho_func.vector().apply('insert')
        self.cp_func.vector().apply('insert')
        
        # Print some debug info
        # avg_k = np.mean(k_values)
        # avg_rho = np.mean(rho_values) 
        # avg_cp = np.mean(cp_values)
        # print(f"Updated properties - avg k: {avg_k:.1f}, avg rho: {avg_rho:.1f}, avg cp: {avg_cp:.1f}")
        min_temp = np.min(temp_values)
        max_temp = np.max(temp_values)
        avg_k = np.mean(k_values)
        avg_rho = np.mean(rho_values)
        avg_cp = np.mean(cp_values)
        
        # Calculate new thermal diffusivity
        avg_alpha = avg_k / (avg_rho * avg_cp)
        
        print(f"TEMP-DEP PROPS: T_range=[{min_temp:.1f}-{max_temp:.1f}]K, "
            f"avg_k={avg_k:.1f}, avg_rho={avg_rho:.1f}, avg_cp={avg_cp:.1f}, "
            f"α={avg_alpha:.2e} m²/s")

    def update_laser_source(self, t_real):
        """Update laser heat source for current time step"""
      

        # Laser is stationary at center of mesh
        L = self.parameters['length']
        W = self.parameters['width']
        H = self.parameters['height']
        
        laser_x = L / 2  # Center of mesh
        laser_y = W / 2  # Center of mesh
        laser_z = H      # Top surface

        laser_x = self.parameters.get('laser_x_position', L / 2)  # Default to center
        laser_y = self.parameters.get('laser_y_position', W / 2)  # Default to center
        laser_z = H  # Keep at top surface
        
        # Validate laser position is within mesh bounds
        if not (0 <= laser_x <= L):
            print(f"Warning: laser_x_position {laser_x} is outside mesh bounds [0, {L}]. Clamping.")
            laser_x = max(0, min(laser_x, L))
        
        if not (0 <= laser_y <= W):
            print(f"Warning: laser_y_position {laser_y} is outside mesh bounds [0, {W}]. Clamping.")
            laser_y = max(0, min(laser_y, W))
        
        # Laser parameters
        scale_factor = self.parameters.get('time_scale_factor', 1.0)

        if self.power_interpolator is not None:
            current_power = self.power_interpolator(t_real)/np.sqrt(scale_factor)
            print(f"DEBUG: Using interpolator at t_real={t_real:.4f}s, power={current_power:.1f}W")
        else:
            current_power = self.parameters['peak_laser_power']/scale_factor
            print(f"DEBUG: Using constant power={current_power:.1f}W")


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
                
        radius = self.parameters['beam_radius']
        absorptivity = self.parameters['absorptivity']
        
        # Create and interpolate laser source
        laser_expr = LaserSource(laser_x, laser_y, laser_z, current_power, radius, absorptivity, degree=2)
        self.laser_source.interpolate(laser_expr)
        
        print(f"Laser at ({laser_x*1000:.2f}, {laser_y*1000:.2f}, {laser_z*1000:.2f}) mm")
        print(f"Mesh bounds: (0, 0, 0) to ({L*1000:.2f}, {W*1000:.2f}, {H*1000:.2f}) mm")
        print(f"Power: {current_power:.1f}W, Radius: {radius*1000:.2f}mm, Absorptivity: {absorptivity}")
        
    def get_laser_position(self, t):
        """Get laser position at time t (stationary at center) Could be changed to move the laser"""
        L = self.parameters['length']
        W = self.parameters['width']
        
        # Use custom positions if provided, otherwise use center
        laser_x = self.parameters.get('laser_x_position', L / 2)
        laser_y = self.parameters.get('laser_y_position', W / 2)

        return laser_x, laser_y


    def _get_custom_equation_properties(self, T):
        """Calculate properties using custom equations"""
        base_vars = {
            'k_base': self.parameters['k_base'],
            'rho_base': self.parameters['rho_base'],
            'cp_base': self.parameters['cp_base'],
        }
        
        # Initialize parsers if not already done
        if not hasattr(self, '_equation_parsers'):
            self._equation_parsers = {}
            
            for prop in ['k', 'rho', 'cp']:
                equation_key = f'{prop}_equation'
                if equation_key in self.parameters:
                    try:
                        self._equation_parsers[prop] = EquationParser(
                            self.parameters[equation_key], 
                            base_vars
                        )
                    except Exception as e:
                        print(f"Warning: Invalid {prop} equation: {e}")
                        # Fall back to constant value
                        self._equation_parsers[prop] = None
        
        # Evaluate equations
        result = {}
        for prop in ['k', 'rho', 'cp']:
            if prop in self._equation_parsers and self._equation_parsers[prop] is not None:
                try:
                    result[prop] = self._equation_parsers[prop].evaluate(T=T)
                except Exception as e:
                    print(f"Warning: Error evaluating {prop} equation at T={T}: {e}")
                    # Fall back to base value
                    result[prop] = base_vars[f'{prop}_base']
            else:
                # Use base value if no equation
                result[prop] = base_vars[f'{prop}_base']
        
        return result

    def get_material_properties(self, temperature):
        """Get material properties at given temperature"""
        if not self.parameters['use_temperature_dependent_properties']:
            return {
                'k': self.parameters['k'],
                'rho': self.parameters['rho'],
                'cp': self.parameters['cp']
            }
        
        variation_type = self.parameters['property_variation_type']
        
        if variation_type == 'linear':
            return self._get_linear_properties(temperature)
        elif variation_type == 'polynomial':
            return self._get_polynomial_properties(temperature)
        elif variation_type == 'piecewise':
            return self._get_piecewise_properties(temperature)
        elif variation_type == 'from_file':
            return self._get_file_properties(temperature)
        elif variation_type == 'custom_equation':
            return self._get_custom_equation_properties(temperature)
        else:
            raise ValueError(f"Unknown property variation type: {variation_type}")

    def run(self, progress_callback=None, use_parallel=True):
        """Run the 3D laser heating simulation with optional parallel processing"""
        if not FENICS_AVAILABLE:
            return self._run_fallback(progress_callback)
        
        rank = 0

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
            
            # Time stepping parameters
            total_time = time_info['total_time']
            base_dt = time_info['dt']
            output_interval = time_info['output_interval']

            # Storage for results - Store RAW (uncorrected) data
            times = []
            max_temperatures = []
            temperature_fields = []
            avg_surface_temperatures = []
            surface_temp_stats_list = []
            laser_spot_max_temps = []
            laser_spot_pyrometer_temps = []

            # Time stepping loop
            t_scaled = 0
            step = 0
            prev_max_temp = self.parameters['T_ambient']

            print(f"SIMULATION SETUP:")
            print(f"  Total time: {self.parameters.get('total_time')} s")
            print(f"  Output interval: {self.parameters.get('output_time_interval', 0.1)} s")
            print(f"  Time scale factor: {self.parameters.get('time_scale_factor', 1.0)}")

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

                self.update_laser_source(t_real)
                
                # Solve time step using time solver
                success, error = self.time_solver.solve_time_step(self.F, self.u, self.bcs, step, t_scaled)
                
                if not success:
                    if rank == 0:
                        print(error)
                    break

                # NO CORRECTION HERE - just update properties if needed
                if self.parameters.get('use_temperature_dependent_properties', False):
                    self.update_material_properties()

                # Update current max temp (RAW, uncorrected)
                current_max_temp = self.u.vector().max()
                
                # Calculate laser spot and surface temperatures (RAW)
                laser_spot_stats = self.get_max_temperature_in_laser_spot()
                laser_spot_max_temp = laser_spot_stats['max_temp']

                surface_temp_stats = self.get_average_surface_temperature_in_laser_spot()
                current_avg_surface_temp = surface_temp_stats['avg_temp']
                laser_spot_pyrometer_temp = self.convert_to_pyrometer_reading(current_avg_surface_temp)

                # Progress reporting
                if step % 5 == 0 or step < 20:
                    if rank == 0:
                        print(f"Step {step:4d}, Real Time: {t_real:.4f}s, "
                                f"Max Temp: {current_max_temp:.1f}K, "
                                f"Avg Surface (laser spot): {current_avg_surface_temp:.1f}K, "
                                f"Pyrometer: {laser_spot_pyrometer_temp:.1f}K")

                # Store RAW (uncorrected) results
                if self.should_output_at_time(t_real, times):
                    # Get RAW temperature values
                    temp_values = self.u.vector().get_local()
                    
                    print(f"STORING RAW DATA: Step {step}, t_real={t_real:.4f}, max_temp={current_max_temp:.1f}K")
                    
                    # Store all RAW data
                    times.append(t_real)
                    max_temperatures.append(current_max_temp)
                    temperature_fields.append(temp_values.tolist())
                    avg_surface_temperatures.append(current_avg_surface_temp)
                    surface_temp_stats_list.append(surface_temp_stats)
                    laser_spot_max_temps.append(laser_spot_max_temp)
                    laser_spot_pyrometer_temps.append(laser_spot_pyrometer_temp)
                    
                    print(f"  Total data points stored: {len(times)}")
                    
                    if progress_callback:
                        real_progress = (t_real / self.parameters['total_time']) * 100
                        progress_callback(real_progress)

                # Update for next step
                self.u_n.assign(self.u)
                prev_max_temp = current_max_temp
                
            print("Time stepping complete. Preparing results...")
            print(f"Total time points stored: {len(times)}")
            
            # Compile RAW results
            raw_results = {
                'times': times,
                'max_temperatures': max_temperatures,
                'temperature_fields': temperature_fields,
                'avg_surface_temperatures': avg_surface_temperatures,
                'surface_temp_stats': surface_temp_stats_list, 
                'laser_spot_max_temperatures': laser_spot_max_temps,
                'laser_spot_pyrometer_temperatures': laser_spot_pyrometer_temps,
                # 'melt_pool_volumes': [0.0] * len(times),  # Placeholder
                # 'heat_fluxes': [0.0] * len(times),  # Placeholder
                'final_temperature_function': self.u.copy(), #deepcopy=Tru
                'parameters': self.parameters.copy()
            }

            # Apply thermal correction post-processing
            # print("Applying thermal corrections...")
            # corrected_results = self.post_process_thermal_correction(raw_results)
            
            print("Simulation and post-processing complete!")
            return raw_results
                
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

    def get_max_temperature_in_laser_spot(self, temp_function=None):
        """
        Calculate maximum temperature on the surface within the laser spot radius
        
        Args:
            temp_function: Temperature function to evaluate (uses self.u if None)
        
        Returns:
            dict: Contains max temperature, location, and other statistics
        """
        if temp_function is None:
            temp_function = self.u
        
        if temp_function is None:
            return {'max_temp': self.parameters['T_ambient'], 'num_points': 0}
        
        # Get laser parameters
        L = self.parameters['length']
        W = self.parameters['width'] 
        H = self.parameters['height']
        laser_radius = self.parameters['beam_radius']
        
        # Laser position (center of surface)
        laser_x = L / 2
        laser_y = W / 2
        laser_z = H  # Top surface
        
        # Create sampling points within the laser spot
        n_radial = 50  # Number of radial divisions
        n_angular = 50  # Number of angular divisions
        
        temperatures = []
        valid_points = 0
        
        # Sample at center point
        try:
            center_point = df.Point(laser_x, laser_y, laser_z)
            center_temp = temp_function(center_point)
            temperatures.append(center_temp)
            valid_points += 1
        except RuntimeError:
            pass
        
        # Sample in concentric circles
        for r_idx in range(1, n_radial + 1):
            radius = (r_idx / n_radial) * laser_radius * 2  # 2x radius for larger sampling area
            
            for theta_idx in range(n_angular):
                theta = (theta_idx / n_angular) * 2 * np.pi
                
                # Calculate point coordinates
                x = laser_x + radius * np.cos(theta)
                y = laser_y + radius * np.sin(theta)
                z = laser_z
                
                # Check if point is within mesh bounds
                if (0 <= x <= L) and (0 <= y <= W):
                    try:
                        point = df.Point(x, y, z)
                        temp = temp_function(point)
                        temperatures.append(temp)
                        valid_points += 1
                    except RuntimeError:
                        # Point outside mesh domain
                        continue
        
        if valid_points == 0:
            return {
                'max_temp': self.parameters['T_ambient'],
                'min_temp': self.parameters['T_ambient'],
                'avg_temp': self.parameters['T_ambient'],
                'num_points': 0,
                'std_temp': 0.0,
                'sampling_radius_mm': laser_radius * 2 * 1000
            }
        
        temperatures = np.array(temperatures)
        
        return {
            'max_temp': np.max(temperatures),
            'min_temp': np.min(temperatures),
            'avg_temp': np.mean(temperatures),
            'num_points': valid_points,
            'std_temp': np.std(temperatures),
            'sampling_radius_mm': laser_radius * 2 * 1000
        }
        
    def convert_to_pyrometer_reading(self, actual_temp, surface_emissivity=None):
        """
        Convert actual temperature to pyrometer reading assuming black body
        
        Args:
            actual_temp: Actual surface temperature (K)
            surface_emissivity: Surface emissivity (uses parameter if None)
        
        Returns:
            pyrometer_temp: Temperature the pyrometer would display (K)
        """
        if surface_emissivity is None:
            surface_emissivity = self.parameters['emissivity']
        
        sigma = self.parameters['stefan_boltzmann']
        T_ambient = self.parameters['T_ambient']
        
        # Calculate actual radiated power per unit area
        actual_radiated_power = surface_emissivity * sigma * (actual_temp**4 - T_ambient**4)
        
        # Calculate what temperature a black body would need to emit this power
        # For black body: P = σ(T^4 - T_ambient^4)
        if actual_radiated_power <= 0:
            return T_ambient
        
        pyrometer_temp_4th = (actual_radiated_power / sigma) + T_ambient**4
        pyrometer_temp = pyrometer_temp_4th**(1/4)
        
        return pyrometer_temp

    def get_average_surface_temperature_in_laser_spot(self, temp_function=None):
        """
        Calculate average temperature on the surface within the laser spot radius
        
        Args:
            temp_function: Temperature function to evaluate (uses self.u if None)
        
        Returns:
            dict: Contains average temperature, number of points, and other statistics
        """
        if temp_function is None:
            temp_function = self.u
        
        if temp_function is None:
            return {'avg_temp': self.parameters['T_ambient'], 'num_points': 0}
        
        # Get laser parameters
        L = self.parameters['length']
        W = self.parameters['width'] 
        H = self.parameters['height']
        laser_radius = self.parameters['beam_radius']
        
        # Laser position (center of surface)
        laser_x = L / 2
        laser_y = W / 2
        laser_z = H  # Top surface
        
        # Create sampling points within the laser spot
        n_radial = 100  # Number of radial divisions
        n_angular = 100  # Number of angular divisions
        
        temperatures = []
        valid_points = 0
        
        # Sample at center point
        try:
            center_point = df.Point(laser_x, laser_y, laser_z)
            center_temp = temp_function(center_point)
            temperatures.append(center_temp)
            valid_points += 1
        except RuntimeError:
            pass
        
        # Sample in concentric circles
        for r_idx in range(1, n_radial + 1):
            radius = (r_idx / n_radial) * laser_radius*10*4 #The two is a multiplier for larger radius
            
            for theta_idx in range(n_angular):
                theta = (theta_idx / n_angular) * 2 * np.pi
                
                # Calculate point coordinates
                x = laser_x + radius * np.cos(theta)
                y = laser_y + radius * np.sin(theta)
                z = laser_z
                
                # Check if point is within mesh bounds
                if (0 <= x <= L) and (0 <= y <= W):
                    try:
                        point = df.Point(x, y, z)
                        temp = temp_function(point)
                        temperatures.append(temp)
                        valid_points += 1
                    except RuntimeError:
                        # Point outside mesh domain
                        continue
        
        if valid_points == 0:
            return {
                'avg_temp': self.parameters['T_ambient'],
                'min_temp': self.parameters['T_ambient'],
                'max_temp': self.parameters['T_ambient'],
                'num_points': 0,
                'std_temp': 0.0,
                'sampling_radius_mm': laser_radius * 20 * 1000  # 2x radius in mm

            }
        
        temperatures = np.array(temperatures)
        
        return {
            'avg_temp': np.mean(temperatures),
            'min_temp': np.min(temperatures),
            'max_temp': np.max(temperatures),
            'num_points': valid_points,
            'std_temp': np.std(temperatures),
            'laser_radius_mm': laser_radius * 1000 *20
        }    
    
    def plot_results(self, results, axes):
        """
        Plots for 3D FEniCS simulation with power scaling.
        Shows surface heat maps, cross-sections, and average laser spot temperatures.
        """
        ax1, ax2, ax3, ax4 = axes
        
        # Clear any existing colorbars
        for cbar in self.colorbars:
            try:
                cbar.remove()
            except:
                pass
        self.colorbars.clear()
        
        # Get final temperature function
        final_temp_func = results.get('final_temperature_function', None)
        
        # Plot 1: Surface temperature heat map
        ax1.clear()
        if final_temp_func is not None:
            try:
                surface_data = self.extract_surface_temperature_from_function(final_temp_func)
                if surface_data:
                    X, Y = surface_data['X'], surface_data['Y']
                    surface_temp = surface_data['temperature']
                    
                    im = ax1.contourf(X*1000, Y*1000, surface_temp, levels=50, cmap='hot')
                    
                    try:
                        cbar = plt.colorbar(im, ax=ax1, label='Temperature (K)', shrink=0.8)
                        self.colorbars.append(cbar)
                    except:
                        pass
                    
                    # Add melting contour if applicable
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
            except Exception as e:
                ax1.set_title('Surface Temperature - Error')
        else:
            ax1.set_title('Surface Temperature - No Temperature Function')
        
        # Plot 2: X-Z Cross-section
        ax2.clear()
        if final_temp_func is not None:
            try:
                self.plot_xz_cross_section_from_function(ax2, final_temp_func)
            except Exception as e:
                ax2.set_title('X-Z Cross-Section - Error')
        else:
            ax2.set_title('X-Z Cross-Section - No Data Available')

        # Plot 3: Y-Z Cross-section
        ax3.clear()
        if final_temp_func is not None:
            try:
                self.plot_yz_cross_section_from_function(ax3, final_temp_func)
            except Exception as e:
                ax3.set_title('Y-Z Cross-Section - Error')
        else:
            ax3.set_title('Y-Z Cross-Section - No Data Available')

        # Plot 4: Temperature vs time - AVERAGE LASER SPOT TEMPERATURES
        ax4.clear()
        
        if 'times' not in results or len(results['times']) == 0:
            ax4.set_title('No Time Data Available')
            return
            
        times = results['times']
        
        # Plot average surface temperatures in laser spot (main curve)
        if 'avg_surface_temperatures' in results and len(results['avg_surface_temperatures']) == len(times):
            ax4.plot(times, results['avg_surface_temperatures'], 'r-', 
                    linewidth=3, label='Average Surface Temp (Laser Spot)')
        
        # Plot pyrometer reading (what you'd actually measure)
        if 'laser_spot_pyrometer_temperatures' in results and len(results['laser_spot_pyrometer_temperatures']) == len(times):
            ax4.plot(times, results['laser_spot_pyrometer_temperatures'], 'b-', 
                    linewidth=2, label='Pyrometer Reading', linestyle='--')
        
        # Plot global max for reference (thinner line)
        if 'max_temperatures' in results and len(results['max_temperatures']) == len(times):
            ax4.plot(times, results['max_temperatures'], 'g-', 
                    linewidth=1.5, label='Global Max Temperature', alpha=0.7)
        
        # Styling
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Temperature (K)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Title with power scaling info
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        if scale_factor > 1:
            ax4.set_title(f'Temperature vs Time (Power Scaled by √{scale_factor:.0f})', fontsize=11)
        else:
            ax4.set_title('Temperature vs Time', fontsize=11)
        
        # Add power info text box
        current_power = self.parameters['peak_laser_power']
        if scale_factor > 1:
            scaled_power = current_power / np.sqrt(scale_factor)
            power_info = f'Original Power: {current_power:.0f}W\nScaled Power: {scaled_power:.1f}W'
        else:
            power_info = f'Laser Power: {current_power:.0f}W'
        
        props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8)
        ax4.text(0.02, 0.98, power_info, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
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
            ax.set_title('X-Z Cross-Section - No Data Available')
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
        try:
            cbar = plt.colorbar(im, ax=ax, label='Temperature (K)', shrink=0.8)
            self.colorbars.append(cbar)  # Store for cleanup
        except Exception as e:
            print(f"Warning: Could not create colorbar for XZ plot: {e}")
        
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
            ax.set_title('Y-Z Cross-Section - No Data Available')
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
        try:
            cbar = plt.colorbar(im, ax=ax, label='Temperature (K)', shrink=0.8)
            self.colorbars.append(cbar)  # Store for cleanup
        except Exception as e:
            print(f"Warning: Could not create colorbar for YZ plot: {e}")
        
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
 