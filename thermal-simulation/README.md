3D Thermal Simulation Framework
A comprehensive 3D thermal simulation framework built with FEniCS for laser heating and thermal analysis applications. Features both GUI and command-line interfaces with advanced capabilities like adaptive mesh refinement, MPI parallelization, and real-time visualization.

Python
FEniCS
License

🚀 Features
Advanced Physics
3D Transient Heat Conduction: Full 3D finite element modeling
Laser Heat Sources: Gaussian beam profiles with realistic physics
Phase Change: Melting and vaporization temperature tracking
Heat Transfer: Conduction, convection, and radiation
Material Properties: Temperature-dependent properties support
Computational Features
Adaptive Mesh Refinement: Automatic mesh refinement near laser spots
MPI Parallelization: Domain decomposition for large simulations
Adaptive Time Stepping: CFL-based and physics-based time step control
Time Scaling: Accelerated physics simulation for faster results
User Interface
Interactive GUI: Point-and-click simulation setup
Real-Time Monitoring: Progress tracking and live output
Visualization: Surface plots, cross-sections, and time series
Batch Processing: Command-line interface for automated runs
📋 Prerequisites
Python 3.8 or higher
Conda package manager (recommended for FEniCS)
🔧 Installation
Step 1: Install FEniCS
Installation Guide: https://me.jhu.edu/nguyenlab/doku.php?id=fenicsx

Conda Installation (Recommended):

# Create a new conda environment
conda create -n fenics python=3.9

# Activate the environment
conda activate fenics

# Install FEniCS from conda-forge
conda install -c conda-forge fenics

# Verify installation
python -c "import dolfin; print('FEniCS installed successfully')"


# Create a new conda environment
conda create -n fenics python=3.9

# Activate the environment
conda activate fenics

# Install FEniCS from conda-forge
conda install -c conda-forge fenics

# Verify installation
python -c "import dolfin; print('FEniCS installed successfully')"
Alternative: Docker Installation

bash
Copy code
# Pull the official FEniCS container
docker pull dolfinx/dolfinx:stable

# Run with GUI support (Linux)
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix dolfinx/dolfinx:stable
Step 2: Install Python Dependencies
bash
Copy code
# Clone the repository
git clone https://github.com/yourusername/thermal-simulation.git
cd thermal-simulation

# Install Python dependencies
pip install -r requirements.txt
🚀 Quick Start
bash
Copy code
python main.py
⚙️ Key Parameters
Note: This version does not take into account temperature dependencies

Material Properties
python
Copy code
{
    'k': 45.0,           # Thermal conductivity (W/m·K)
    'rho': 7850.0,       # Density (kg/m³)
    'cp': 460.0,         # Specific heat (J/kg·K)
    'T_melt': 1811.0,    # Melting temperature (K)
    'T_ambient': 298.0,  # Ambient temperature (K)
}
Laser Parameters
python
Copy code
{
    'laser_power': 1000,     # Laser power (W)
    'beam_radius': 0.001,    # Beam radius (m)
    'laser_speed': 0.01,     # Scanning speed (m/s)
    'absorption': 0.3,       # Material absorption coefficient
}
Simulation Settings
python
Copy code
{
    'total_time': 1.0,           # Total simulation time (s)
    'dt': 0.001,                 # Time step (s)
    'mesh_resolution': 32,       # Base mesh resolution
    'adaptive_refinement': True, # Enable adaptive mesh refinement
    'time_scale_factor': 1.0,    # Time acceleration factor
}
📁 Project Structure
python
Copy code
thermal-simulation/
├── main.py                  # Entry point (GUI/batch modes)
├── gui_app.py              # Tkinter-based GUI application
├── simulation/             # Core simulation package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Base simulation class
│   ├── fixed_temp.py       # Fixed temperature simulations
│   ├── laser_heating.py    # 3D laser heating simulations
│   ├── Mesh.py             # Mesh handling and refinement
│   └── timestep.py         # Time stepping utilities
├── examples/               # Usage examples
├── docs/                   # Documentation
└── tests/                  # Unit tests (if available)
📊 Simulation Types
1. Laser Heating Simulation (LaserHeatingSimulation)
Moving laser heat source with Gaussian beam profile
Beer-Lambert depth attenuation
Radiation and convection boundary conditions
Adaptive time stepping based on power profiles
Melt pool tracking and analysis
Support for time-varying power profiles
2. Fixed Temperature Simulation (FixedTempSimulation)
Fixed temperature boundary conditions
Convective heat transfer modeling
1D/2D simulations for validation
Educational and benchmarking purposes
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Konrad Muly

Institution: Johns Hopkins University
🙏 Acknowledgments
FEniCS Project for the finite element framework
Johns Hopkins University for computational resources
📚 Citation
If you use this software in your research, please cite:

bibtex
Copy code
@software{thermal_simulation_framework,
    title={3D Thermal Simulation Framework with FEniCS},
    author={Konrad Muly},
    year={2024},
    url={https://github.com/yourusername/thermal-simulation}
}
🐛 Known Issues
FEniCS installation can be complex on some systems - use conda when possible
Large 3D meshes may require significant memory (>8GB RAM recommended)
MPI parallelization requires proper MPI installation
Temperature-dependent material properties not yet implemented
📈 Performance Tips
Use time_scale_factor > 1 for faster preliminary results
Enable adaptive refinement only when needed (computationally expensive)
For large simulations, consider using MPI: mpirun -n 4 python main.py