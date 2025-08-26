import dolfin as df
import numpy as np

class TimeSteppingSolver:
    """Handles time stepping and solver configuration for thermal simulations"""
    
    def __init__(self, parameters, function_space, mesh_handler):
        self.parameters = parameters
        self.V = function_space
        self.mesh_handler = mesh_handler
        self.solver_params = self._setup_solver_parameters()
        self.dt_param = None  # Will be set when needed
        
    def _setup_solver_parameters(self):
        """Setup nonlinear solver parameters"""
        solver_params = {
            'nonlinear_solver': 'newton',
            'newton_solver': {
                'linear_solver': 'lu',
                'absolute_tolerance': 1e-8,
                'relative_tolerance': 1e-7,
                'maximum_iterations': 25,
                'relaxation_parameter': 1.0,
                'report': True
            }
        }
        
        # Update with user parameters if provided
        if 'solver_params' in self.parameters:
            solver_params.update(self.parameters['solver_params'])
            
        return solver_params
    
    def check_stability(self):
        """Check if user time step is stable"""
        dt_stable = self.mesh_handler.calculate_stable_time_step()
        dt_user = self.parameters['dt']
        
        if dt_user > dt_stable:
            print(f"WARNING: User dt ({dt_user:.2e}) exceeds stable dt ({dt_stable:.2e})")
            print(f"Simulation may be unstable. Recommend dt ≤ {dt_stable:.2e}")
            
            # Optionally auto-adjust
            if self.parameters.get('auto_adjust_dt', False):
                self.parameters['dt'] = dt_stable
                print(f"Auto-adjusted dt to {dt_stable:.2e}")
                
        return dt_stable
    
    def calculate_power_based_timestep(self, t, current_dt, power_interpolator=None):
        """Calculate time step based on power profile changes"""
        if power_interpolator is None:
            return current_dt
        
        # Get current and next power values
        try:
            current_power = power_interpolator(t)
            
            # Look ahead to see power change rate
            dt_probe = min(current_dt, 0.001)
            next_power = power_interpolator(t + dt_probe)
            
            # Calculate power change rate (W/s)
            power_change_rate = abs(next_power - current_power) / dt_probe
            
            # Adaptive criteria based on power changes
            if power_change_rate > 1000:
                return 1e-4   # 0.1ms
            elif power_change_rate > 100:
                return 5e-4   # 0.5ms
            elif power_change_rate > 10:
                return 1e-3   # 1ms
            elif power_change_rate > 1:
                return 5e-2   # 5ms
            else:
                return 5e-1   # 10ms (maximum)
        except:
            # If power interpolation fails, return current dt
            return current_dt

    def calculate_optimal_timestep(self, t, current_temp, prev_temp, base_dt, power_interpolator=None):
        """Calculate optimal time step with time scaling awareness"""
        scale_factor = self.parameters.get('time_scale_factor', 1.0)
        
        if scale_factor > 10:
            # For scaled simulations
            dt_power = self.calculate_power_based_timestep(t, base_dt, power_interpolator)
            dt_cfl_raw = self.mesh_handler.calculate_stable_time_step()
            dt_cfl_relaxed = dt_cfl_raw * min(100, scale_factor * 0.1)
            
            if base_dt > 0:
                temp_change_rate = abs(current_temp - prev_temp) / base_dt
                if temp_change_rate > 500:
                    dt_temp = 200 / max(temp_change_rate, 1e-6)
                else:
                    dt_temp = base_dt * 10
            else:
                dt_temp = base_dt
            
            dt_optimal = min(dt_power, dt_cfl_relaxed, dt_temp)
            result = max(base_dt * 0.1, min(dt_optimal, base_dt * 50))
            
            if t < 0.001:  # Debug for first few steps
                print(f"  dt_power: {dt_power:.2e}")
                print(f"  dt_cfl_raw: {dt_cfl_raw:.2e}")
                print(f"  dt_cfl_relaxed: {dt_cfl_relaxed:.2e}")
                print(f"  dt_temp: {dt_temp:.2e}")
                print(f"  base_dt: {base_dt:.2e}")
                print(f"  chosen dt: {result:.2e}")
            
            return result
        else:
            # Original logic for unscaled simulations
            dt_power = self.calculate_power_based_timestep(t, base_dt, power_interpolator)
            dt_cfl = self.mesh_handler.calculate_stable_time_step()
            
            if base_dt > 0:
                temp_change_rate = abs(current_temp - prev_temp) / base_dt
                if temp_change_rate > 100:
                    dt_temp = 50 / max(temp_change_rate, 1e-6)
                else:
                    dt_temp = base_dt * 1.5
            else:
                dt_temp = base_dt
            
            dt_optimal = min(dt_power, dt_cfl, dt_temp)
            return max(1e-6, min(dt_optimal, 0.01))

    def update_dt(self, new_dt):
        """Update the time step parameter"""
        if self.dt_param is not None:
            self.dt_param.assign(new_dt)
        else:
            # Store for later use
            self.current_dt = new_dt
    
    def set_dt_param(self, dt_param):
        """Set the FEniCS Constant for time step"""
        self.dt_param = dt_param
    
    def solve_time_step(self, F, u, bc, step, t):
        """Solve a single time step"""
        try:
            from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
            
            J = df.derivative(F, u)
            problem = NonlinearVariationalProblem(F, u, bc, J)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(self.solver_params)
            solver.solve()
            
            return True, None
            
        except Exception as e:
            error_msg = f"Solver failed at step {step}, time {t:.3f}: {e}"
            return False, error_msg
    
    def should_refine_mesh(self, step):
        """Check if mesh should be refined at this step"""
        refinement_interval = self.parameters.get('refinement_interval', 10)
        return step % refinement_interval == 0 and step > 0
    
    # def should_output_results(self, step):
    #     """Check if results should be stored at this step"""
    #     output_interval = self.parameters.get('output_interval', 10)
    #     return step % output_interval == 0
    
    # def should_output_results(self, step, t_real=None):
    #     """Check if results should be output based on real time intervals"""
    #     output_time_interval = self.parameters.get('output_time_interval', 0.5)
        
    #     # Initialize last output time if not exists
    #     if not hasattr(self, '_last_output_time'):
    #         self._last_output_time = -1.0
        
    #     # Always output first step or when time interval reached
    #     if step == 0 or (t_real - self._last_output_time >= output_time_interval):
    #         self._last_output_time = t_real
    #         return True
        
    #     return False
    def should_output_results_by_time(self, current_time, existing_times, output_interval):
        """Time-based output decision"""
        # Always output at t=0
        if len(existing_times) == 0:
            return True
        
        # Check if enough time has passed since last output
        last_output_time = existing_times[-1] if existing_times else 0
        return (current_time - last_output_time) >= output_interval

    def should_output_results(self, step, t_real):
        """Keep existing step-based method for backward compatibility"""
        output_interval = self.parameters.get('output_interval', 10)
        return step % output_interval == 0


    def get_time_stepping_info(self):
        """Get time stepping parameters"""
        total_time = self.parameters['total_time']
        dt = self.parameters['dt']
        n_steps = int(total_time / dt)
        
        return {
            'total_time': total_time,
            'dt': dt,
            'n_steps': n_steps,
            'output_interval': self.parameters.get('output_interval', 10)
        }