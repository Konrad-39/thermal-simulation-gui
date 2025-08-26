"""
Fixed temperature boundary condition simulation.
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dolfin import *
import dolfin as df
from .base import SimulationBase  # Add this import

class FixedTempSimulation(SimulationBase):  # Add inheritance here
    """Simulation with fixed temperature boundary condition"""
    
    def __init__(self):
        super().__init__()  # Call parent constructor
        self.default_parameters = {
            # Material properties
            'k': 45.0,           # Thermal conductivity W/m·K
            'rho': 7850.0,       # Density kg/m³
            'cp': 460.0,         # Specific heat J/kg·K
            'T_ambient': 298.0,  # Ambient temperature K
            
            # Geometry
            'length': 0.01,      # m
            'width': 0.01,       # m
            'height': 0.002,     # m
            'mesh_resolution': 32,
            
            # Boundary condition
            'fixed_temp': 800.0,        # K
            'boundary_location': 'left',  # top, bottom, left, right
            'convection_coeff': 10.0,   # Convection coefficient W/m²·K
            
            # Simulation
            'total_time': 10.0,     # s
            'dt': 0.01,           # s
            'output_interval': 10, # steps
            'output_time_interval': 0.5  # ADD THIS LINE - time-based output interval

        }
        self.parameters = self.default_parameters.copy()

        self.mesh = None
        self.V = None
        self.u = None
        self.u_n = None
        # Remove self.stop_requested since it's in parent class

    def should_output_at_time(self, current_time, existing_times):
        """Determine if we should output results at current time based on time interval"""
        output_interval = self.parameters.get('output_time_interval', 0.5)
        
        # Always output at t=0
        if len(existing_times) == 0:
            return True
        
        # Check if enough time has passed since last output
        last_output_time = existing_times[-1] if existing_times else 0
        return (current_time - last_output_time) >= output_interval

    def setup_simulation(self):
        """Setup FEniCS problem"""
        try:
            # Mesh setup
            nx = self.parameters['mesh_resolution']
            L = self.parameters['length']
            self.mesh = df.IntervalMesh(nx, 0, L)

            # Define function space
            self.V = df.FunctionSpace(self.mesh, 'P', 1)

            # Material properties
            k = self.parameters['k']
            rho = self.parameters['rho']
            cp = self.parameters['cp']
            dt = self.parameters['dt']
            
            # Boundary condition parameters
            T_fixed = self.parameters['fixed_temp']
            T_ambient = self.parameters['T_ambient']
            h_conv = self.parameters['convection_coeff']
            boundary_loc = self.parameters['boundary_location']

            # Create boundary markers
            boundaries = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
            boundaries.set_all(0)

            # Define and mark boundaries
            class LeftBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[0], 0)

            class RightBoundary(df.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and df.near(x[0], L)

            left_bound = LeftBoundary()
            right_bound = RightBoundary()
            
            left_bound.mark(boundaries, 1)   # Left boundary = 1
            right_bound.mark(boundaries, 2)  # Right boundary = 2

            # Create measure for specific boundaries
            ds = df.Measure('ds', domain=self.mesh, subdomain_data=boundaries)

            # Apply fixed temperature boundary condition
            if boundary_loc == 'left':
                # Fixed temp on left boundary
                def left_boundary(x, on_boundary):
                    return on_boundary and df.near(x[0], 0)
                self.bc = df.DirichletBC(self.V, df.Constant(T_fixed), left_boundary)
                convection_boundary = 2  # Apply convection to right boundary
            else: 
                # Fixed temp on right boundary
                def right_boundary(x, on_boundary):
                    return on_boundary and df.near(x[0], L)
                self.bc = df.DirichletBC(self.V, df.Constant(T_fixed), right_boundary)
                convection_boundary = 1  # Apply convection to left boundary

            # Define test function and solution functions
            v = df.TestFunction(self.V)
            self.u = df.Function(self.V)
            self.u.assign(df.Constant(T_ambient))
            
            # Previous time step solution
            self.u_n = df.Function(self.V)
            self.u_n.assign(df.Constant(T_ambient))

            # Nonlinear variational form F(u) = 0
            F = (rho * cp * (self.u - self.u_n) * v + 
                dt * k * df.dot(df.grad(self.u), df.grad(v))) * df.dx

            # Add convective boundary term ONLY to the opposite boundary
            F += dt * h_conv * (self.u - T_ambient) * v * ds(convection_boundary)
            
            # Store the nonlinear form
            self.F = F

            # Solver parameters
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

    def run(self, progress_callback=None):
        """Run the fixed temperature simulation"""
        errors = self.validate_parameters()
        if errors:
            raise ValueError(f"Parameter validation failed: {', '.join(errors)}")
        
        print("Starting FEniCS fixed temperature simulation...")

        # Setup FEniCS problem
        self.setup_simulation()

        # Time stepping params
        total_time = self.parameters['total_time']
        dt = self.parameters['dt']
        output_interval = self.parameters['output_interval']
        n_steps = int(total_time / dt)

        # Storage for Results
        times = []
        temperatures = []
        heat_fluxes = []

        # Get coordinates for plotting
        coordinates = self.V.tabulate_dof_coordinates()
        x_coords = coordinates[:, 0]
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]

        # Time stepping loop
        t = 0
        for step in range(n_steps):
            if self.stop_requested:
                print("Simulation stopped by user")
                break
                
            t += dt
            
            # Try nonlinear variational problem
            # df.solve(self.a == self.L, self.u, self.bc)
            try:
                J = df.derivative(self.F, self.u)
                
                problem = NonlinearVariationalProblem(self.F, self.u, self.bc, J)
                solver = NonlinearVariationalSolver(problem)
                solver.parameters.update(self.solver_params)
                solver.solve()
            except Exception as e:
                print(f"Solver failed at step {step}, time{t:.3f}: with error: {str(e)}")
                break
            
            # Store results at output intervals
            # if step % output_interval == 0:
            #     times.append(t)
                
            #     # Get temperature values
            #     u_values = self.u.vector().get_local()
            #     u_values_sorted = u_values[sort_idx]
            #     temperatures.append(u_values_sorted.copy())
                
            #     # Calculate heat flux (simplified)
            #     max_temp = np.max(u_values)

            #     k = self.parameters['k']

            #     # Get temperature gradient
            #     grad_u = df.project(df.grad(self.u), df.VectorFunctionSpace(self.mesh, 'P', 1))
            #     grad_values = grad_u.vector().get_local()

            #     # Heat flux = -k * gradient (in 1D, just the x-component)
            #     boundary_loc = self.parameters['boundary_location']
            #     if boundary_loc == 'left':
            #         # Heat flux at left boundary (first node)
            #         heat_flux = -k * grad_values[0]
            #     else:
            #         # Heat flux at right boundary (last node)  
            #         heat_flux = -k * grad_values[-1]

            #     heat_fluxes.append(heat_flux)

            #     # Update progress
            #     if progress_callback:
            #         progress = (step / n_steps) * 100
            #         progress_callback(progress)
            
            # Store results based on time interval
            if self.should_output_at_time(t, times):
                times.append(t)
                
                # Get temperature values
                u_values = self.u.vector().get_local()
                u_values_sorted = u_values[sort_idx]
                temperatures.append(u_values_sorted.copy())
                
                # Calculate heat flux (simplified)
                max_temp = np.max(u_values)

                k = self.parameters['k']

                # Get temperature gradient
                grad_u = df.project(df.grad(self.u), df.VectorFunctionSpace(self.mesh, 'P', 1))
                grad_values = grad_u.vector().get_local()

                # Heat flux = -k * gradient (in 1D, just the x-component)
                boundary_loc = self.parameters['boundary_location']
                if boundary_loc == 'left':
                    # Heat flux at left boundary (first node)
                    heat_flux = -k * grad_values[0]
                else:
                    # Heat flux at right boundary (last node)  
                    heat_flux = -k * grad_values[-1]

                heat_fluxes.append(heat_flux)

                # Update progress
                if progress_callback:
                    progress = (t / total_time) * 100  # Change to time-based progress
                    progress_callback(progress)


            # Update previous solution
            self.u_n.assign(self.u)

        # Final progress update
        if progress_callback:
            progress_callback(100.0)

        # Prepare results
        results = {
            'times': np.array(times),
            'temperatures': np.array(temperatures),
            'heat_fluxes': np.array(heat_fluxes),
            'x_coordinates': x_coords,
            'final_temperature': temperatures[-1] if temperatures else None
        }

        return results

    def plot_results(self, results, axes):
        """Plot simulation results"""
        if not results:
            return
            
        ax1, ax2, ax3, ax4 = axes
        
        # Clear axes
        for ax in axes:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        times = results.get('times', [])
        temperatures = results.get('temperatures', [])
        heat_fluxes = results.get('heat_fluxes', [])
        x_coords = results.get('x_coordinates', [])
        
        if len(temperatures) == 0 or len(times) == 0:
            for i, ax in enumerate(axes):
                ax.set_title(f"Plot {i+1} - No Data")
            return
        
        try:
            # Plot 1: Final temperature distribution
            final_temp = temperatures[-1]
            ax1.plot(x_coords, final_temp, 'r-', linewidth=2)
            ax1.set_title("Final Temperature Distribution")
            ax1.set_xlabel("Position (m)")
            ax1.set_ylabel("Temperature (K)")
        except Exception as e:
            ax1.set_title("Final Temperature Distribution - Error")
            print(f"Plot 1 error: {e}")
        
        try:
            # Plot 2: Maximum temperature vs time
            max_temps = [np.max(temp) for temp in temperatures]
            ax2.plot(times, max_temps, 'b-', linewidth=2)
            ax2.set_title("Maximum Temperature vs Time")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Temperature (K)")
        except Exception as e:
            ax2.set_title("Max Temperature vs Time - Error")
            print(f"Plot 2 error: {e}")
        
        try:
            # Plot 3: Maximum temperature evolution
            ax3.plot(times, heat_fluxes, 'g-', linewidth=2)
            ax3.set_title("Heat Flux at Boundary")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Heat Flux (W/m²)")
        except Exception as e:
            ax3.set_title("Temperature Evolution - Error")
            print(f"Plot 3 error: {e}")
        
        try:
            # Plot 4: Temperature at different locations
            if len(x_coords) >= 4:
                quarter_idx = len(x_coords) // 4
                half_idx = len(x_coords) // 2
                three_quart_idx = 3 * len(x_coords) // 4
                
                temp_matrix = np.array(temperatures)
                ax4.plot(times, temp_matrix[:, quarter_idx], 'g-', 
                        label=f'x = {x_coords[quarter_idx]:.3f} m')
                ax4.plot(times, temp_matrix[:, half_idx], 'b-', 
                        label=f'x = {x_coords[half_idx]:.3f} m')
                ax4.plot(times, temp_matrix[:, three_quart_idx], 'r-', 
                        label=f'x = {x_coords[three_quart_idx]:.3f} m')
                
                ax4.set_title("Temperature vs Time at Different Locations")
                ax4.set_xlabel("Time (s)")
                ax4.set_ylabel("Temperature (K)")
                ax4.legend()
            else:
                ax4.set_title("Temperature at Different Locations - Insufficient Points")
        except Exception as e:
            ax4.set_title("Temperature at Locations - Error")
            print(f"Plot 4 error: {e}")

    def validate_parameters(self):
        """Validate parameters specific to fixed temp simulation"""
        errors = []

        # Check required params 
        required = ['k', 'rho', 'cp', 'T_ambient', 'fixed_temp', 
                   'length', 'total_time', 'dt', 'mesh_resolution']

        for param in required:
            if param not in self.parameters:
                errors.append(f"Missing parameter: {param}")

        # Check positive parameters
        positive_params = ['k', 'rho', 'cp', 'length', 'total_time', 'dt', 'mesh_resolution']
        for param in positive_params:
            if param in self.parameters and self.parameters[param] <= 0:
                errors.append(f"Parameter {param} must be positive")

        # Check temperature parameters
        if 'T_ambient' in self.parameters and self.parameters['T_ambient'] <= 0:
            errors.append("Ambient temperature must be positive (Kelvin)")
        
        if 'fixed_temp' in self.parameters and self.parameters['fixed_temp'] <= 0:
            errors.append("Fixed temperature must be positive (Kelvin)")

        # Check boundary location
        if 'boundary_location' in self.parameters:
            valid_locations = ['left', 'right', 'top', 'bottom']
            if self.parameters['boundary_location'] not in valid_locations:
                errors.append(f"Boundary location must be one of: {valid_locations}")

        return errors