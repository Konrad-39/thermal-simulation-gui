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
        """Save results to file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__class__') and 'dolfin' in str(obj.__class__):
                return "FEniCS_Function_object_Not_Serializable"
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
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