# simulation/__init__.py
"""
Thermal simulation package.

This package contains different simulation types for thermal analysis.
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Konrad Muly"
__description__ = "Thermal simulation package using FEniCS"

# Common simulation parameters and utilities
DEFAULT_MATERIAL_PROPERTIES = {
    'thermal_conductivity': 45.0,  # W/m·K
    'density': 7850.0,            # kg/m³
    'specific_heat': 460.0,       # J/kg·K
    'melting_temp': 1811.0,       # K
    'ambient_temp': 298.0,        # K
}

DEFAULT_GEOMETRY = {
    'length': 0.01,      # m
    'width': 0.01,       # m
    'height': 0.002,     # m
    'mesh_resolution': 32
}

DEFAULT_SIMULATION_SETTINGS = {
    'total_time': 1.0,      # s
    'time_step': 0.01,      # s
    'output_interval': 10,   # steps
    'solver_tolerance': 1e-6
}

# Import simulation classes (after defining constants)
from .base import SimulationBase
from .fixed_temp import FixedTempSimulation
from .laser_heating import LaserHeatingSimulation

__all__ = ['SimulationBase', 'FixedTempSimulation', 'LaserHeatingSimulation']

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    try:
        import dolfin
    except ImportError:
        missing.append("FEniCS (dolfin)")
    
    try:
        import numpy
    except ImportError:
        missing.append("NumPy")
        
    try:
        import matplotlib
    except ImportError:
        missing.append("Matplotlib")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install with: pip install fenics numpy matplotlib")
        return False
    return True