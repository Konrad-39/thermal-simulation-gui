#!/usr/bin/env env python3
"""
Main entry point for Thermal Simulation application.
Can run GUI or batch simulations.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_gui():
    """Launch the GUI application"""
    try:
        import dolfin
        print("FEniCS loaded succesfully")

        import tkinter as tk
        from gui_app import ThermalSimulationGUI
        
        root = tk.Tk()
        app = ThermalSimulationGUI(root)
        root.mainloop()
        
    except ImportError as e:
        if "dolfin" in str(e):
            print("FEniCS not found. Please install FEniCS first.")
        else:
            print(f"Error importing components: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error launching GUI: {e}")
        sys.exit(1)

def run_batch_simulation(sim_type, config_file=None):
    """Run simulation in batch mode"""
    try:
        if sim_type == "fixed_temp":
            from simulation.fixed_temp import FixedTempSimulation
            sim = FixedTempSimulation()
        elif sim_type == "laser_heating":
            from simulation.laser_heating import LaserHeatingSimulation
            sim = LaserHeatingSimulation()
        else:
            raise ValueError(f"Unknown simulation type: {sim_type}")
        
        if config_file:
            sim.load_config(config_file)
        
        print(f"Running {sim_type} simulation...")
        results = sim.run()
        print("Simulation completed successfully")
        
        # Save results
        output_file = f"{sim_type}_results.json"
        sim.save_results(results, output_file)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error running batch simulation: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Thermal Simulation Application")
    parser.add_argument("--mode", choices=["gui", "batch"], default="gui",
                       help="Run mode: gui (default) or batch")
    parser.add_argument("--sim-type", choices=["fixed_temp", "laser_heating"],
                       help="Simulation type for batch mode")
    parser.add_argument("--config", type=str,
                       help="Configuration file for batch mode")
    
    args = parser.parse_args()
    
    if args.mode == "gui":
        print("Launching GUI...")
        run_gui()
    elif args.mode == "batch":
        if not args.sim_type:
            print("Error: --sim-type required for batch mode")
            sys.exit(1)
        run_batch_simulation(args.sim_type, args.config)
    
if __name__ == "__main__":
    main()