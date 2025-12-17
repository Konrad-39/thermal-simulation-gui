# simulation/base.py
"""
Base class for all thermal simulations
"""
import json
import numpy as np

class SimulationBase:
    """Base class for all thermal simulations"""
    
    def __init__(self):
        self.parameters = {}
        self.results = None
        self.stop_requested = False
        
    def set_parameters(self, params):
        """Set simulation parameters"""
        self.parameters.update(params)
        
    def get_parameters(self):
        """Get current parameters"""
        return self.parameters.copy()
        
    def validate_parameters(self):
        """Validate simulation parameters"""
        errors = []
        # Override in subclasses
        return errors
        
    def stop(self):
        """Request simulation stop"""
        self.stop_requested = True
        
    def run(self, progress_callback=None):
        """Run simulation - override in subclasses"""
        raise NotImplementedError("Subclasses must implement run method")
        
    def plot_results(self, results, axes):
        """Plot results - override in subclasses"""
        raise NotImplementedError("Subclasses must implement plot_results method")
        
    def save_results(self, results, filename):
        # """Save results to file"""
        # # Convert numpy arrays to lists for JSON serialization
        # def convert_numpy(obj):
        #     if isinstance(obj, np.ndarray):
        #         return obj.tolist()
        #     elif hasattr(obj, '__class__') and 'dolfin' in str(obj.__class__):
        #         return "FEniCS_Function_object_Not_Serializable"
        #     elif isinstance(obj, dict):
        #         return {k: convert_numpy(v) for k, v in obj.items()}
        #     elif isinstance(obj, list):
        #         return [convert_numpy(item) for item in obj]
        #     else:
        #         return obj
        
        # serializable_results = convert_numpy(results)
        
        # with open(filename, 'w') as f:
        #     json.dump({
        #         'parameters': self.parameters,
        #         'results': serializable_results
        #     }, f, indent=2)
        # def save_results(self, results, filename):
        """Save results to file"""
        # Convert numpy arrays and FEniCS objects to serializable format
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__class__') and 'dolfin' in str(obj.__class__):
                # Skip FEniCS function objects
                return None
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Remove non-serializable items
        serializable_results = {}
        for key, value in results.items():
            if key == 'final_temperature_function':
                continue  # Skip FEniCS function
            converted = convert_for_json(value)
            if converted is not None:
                serializable_results[key] = converted
        
        with open(filename, 'w') as f:
            json.dump({
                'parameters': self.parameters,
                'results': serializable_results
            }, f, indent=2)
            
    def load_config(self, filename):
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        if 'parameters' in config:
            self.set_parameters(config['parameters'])
            
    def cleanup(self):
        """Clean up resources to free memory"""
        if hasattr(self, 'mesh'):
            del self.mesh
        if hasattr(self, 'V'):
            del self.V
        if hasattr(self, 'u'):
            del self.u
        if hasattr(self, 'u_n'):
            del self.u_n

    # Add this method to SimulationBase:

    def verify_solver_consistency(self, other_simulation):
        """Verify that two simulations use the same solver parameters"""
        my_params = self.get_default_solver_params()
        other_params = other_simulation.get_default_solver_params()
        
        def compare_dicts(d1, d2, path=""):
            differences = []
            for key in set(d1.keys()) | set(d2.keys()):
                if key not in d1:
                    differences.append(f"{path}.{key}: missing in first")
                elif key not in d2:
                    differences.append(f"{path}.{key}: missing in second")
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    differences.extend(compare_dicts(d1[key], d2[key], f"{path}.{key}"))
                elif d1[key] != d2[key]:
                    differences.append(f"{path}.{key}: {d1[key]} != {d2[key]}")
            return differences
        
        differences = compare_dicts(my_params, other_params)
        
        if differences:
            print("Solver parameter differences found:")
            for diff in differences:
                print(f"  {diff}")
            return False
        else:
            print("✓ Solver parameters are consistent")
            return True

    @staticmethod
    def get_default_solver_params():
        """Get default solver parameters used across all simulations"""
        return {'nonlinear_solver': 'newton',
                'newton_solver': {
                    'linear_solver': 'mumps',
                    'relative_tolerance': 1e-9,
                    'absolute_tolerance': 1e-10,
                    'maximum_iterations': 100,
                    'preconditioner': 'default',
                    'relaxation_parameter': 0.5,
                    'error_on_nonconvergence': True,
                    'convergence_criterion': 'incremental'
                }
            }