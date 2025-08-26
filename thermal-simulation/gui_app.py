"""
Main GUI application for thermal simulation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
import time
from pathlib import Path
# Import simulation classes
from simulation.fixed_temp import FixedTempSimulation
from simulation.laser_heating import LaserHeatingSimulation
from simulation import check_dependencies

class ThermalSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Simulation GUI")
        self.root.geometry("1400x900")
        
        # Check Dependencies
        if not check_dependencies():
            messagebox.showerror("Dependencies Missing",
                "Required dependencies are missing. Please install FEniCS and other requirements.")
            return

        # Simulation state
        self.current_simulation = None
        self.simulation_running = False
        self.simulation_thread = None
        self.results = None
        
        # Add these timer variables:
        self.start_time = None
        self.timer_var = tk.StringVar(value="00:00:00")
        self.timer_job = None  # For scheduling timer updates

        # GUI variables
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.sim_type_var = tk.StringVar(value="fixed_temp")
        
        # Parameter storage
        self.param_vars = {}
        
        self.create_gui()

    def create_gui(self):
        """Create the main GUI layout"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for plots
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Create control panel
        self.create_control_panel(left_frame)
        
        # Create plot area
        self.create_plot_area(right_frame)

    def create_control_panel(self, parent):
        """Create the control panel with parameters"""
        # Simulation type selection
        type_frame = ttk.LabelFrame(parent, text="Simulation Type")
        type_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(type_frame, text="Fixed Temperature", 
                       variable=self.sim_type_var, value="fixed_temp",
                       command=self.on_sim_type_change).pack(anchor='w', padx=5, pady=2)
        ttk.Radiobutton(type_frame, text="Laser Heating", 
                       variable=self.sim_type_var, value="laser_heating",
                       command=self.on_sim_type_change).pack(anchor='w', padx=5, pady=2)
        
        # Parameter notebook
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create parameter tabs
        self.create_material_tab()
        self.create_geometry_tab()
        self.create_boundary_tab()
        self.create_simulation_tab()
        
        # Control buttons
        self.create_control_buttons(parent)
        
        # Status and progress
        self.create_status_panel(parent)

    def create_material_tab(self):
        """Create material properties tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Material")
        
        #Default material properties
        material_params = [
            ("Thermal Conductivity (W/m·K)", "k", 170.0),
            ("Density (kg/m³)", "rho", 3200.0),
            ("Specific Heat (J/kg·K)", "cp", 510.0),
            ("Melting Temperature (K)", "T_melt", 3103.0),
            ("Ambient Temperature (K)", "T_ambient", 298.0),
            ("Emissivity", "emissivity", 0.8),  # Added emissivity
            ("Stefan-Boltzmann Constant", "stefan_boltzmann", 5.67e-8)
        ]
        
        for i, (label, key, default) in enumerate(material_params):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)

            if key == "stefan_boltzmann":
            # Make Stefan-Boltzmann constant read-only (it's a physical constant)
                var = tk.DoubleVar(value=default)
                entry = ttk.Entry(frame, textvariable=var, width=15, state='readonly')
                ttk.Label(frame, text="(Physical constant)", font=('TkDefaultFont', 8, 'italic')).grid(
                    row=i, column=2, sticky='w', padx=5)
            else:
                var = tk.DoubleVar(value=default)
                entry = ttk.Entry(frame, textvariable=var, width=15)

            entry.grid(row=i, column=1, padx=5, pady=5)
            self.param_vars[key] = var

    def start_timer(self):
        """Start the simulation timer"""
        import time
        self.start_time = time.time()
        self.update_timer()

    def stop_timer(self):
        """Stop the simulation timer"""
        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

    def update_timer(self):
        """Update the timer display"""
        if self.start_time is not None and self.simulation_running:
            import time
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.timer_job = self.root.after(1000, self.update_timer)

    def create_geometry_tab(self):
        """Create geometry parameters tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Geometry")
        
        geometry_params = [
            ("Length (m)", "length", 0.008),
            ("Width (m)", "width", 0.008),
            ("Height (m)", "height", 0.0005),
            ("Mesh Resolution", "mesh_resolution", 10),
        ]
        
        for i, (label, key, default) in enumerate(geometry_params):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            if key == "mesh_resolution":
                var = tk.IntVar(value=default)
            else:
                var = tk.DoubleVar(value=default)
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.param_vars[key] = var

    def create_boundary_tab(self):
        """Create boundary conditions tab"""
        self.boundary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.boundary_frame, text="Boundary")
        self.update_boundary_tab()

    def create_simulation_tab(self):
        """Create simulation parameters tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Simulation")
        
        # Basic simulation parameters
        sim_params = [
            ("Total Time (s)", "total_time", 10.0),
            ("Time Step (s)", "dt", 0.01),
            ("Output Interval", "output_interval", 10),
            ("Convection Coefficient (W/m²·K)", "convection_coeff", 0.0),
        ]

        for i, (label, key, default) in enumerate(sim_params):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            if key == "output_interval":
                var = tk.IntVar(value=default)
            else:
                var = tk.DoubleVar(value=default)
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.param_vars[key] = var

        # TIME SCALING SECTION
        row = len(sim_params)
        
        # Add separator
        separator = ttk.Separator(frame, orient='horizontal')
        separator.grid(row=row, column=0, columnspan=3, sticky='ew', padx=5, pady=10)
        
        row += 1
        ttk.Label(frame, text="Time Scaling", 
                font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, 
                                                        sticky='w', padx=5, pady=5)
        
        time_scaling_params = [
            ("Enable Time Scaling", "scale_thermal_properties", False, 'bool'),
            ("Time Scale Factor", "time_scale_factor", 1.0, 'float'),
            ("Auto-adjust Time Step", "auto_adjust_dt", False, 'bool'),
        ]
        
        for i, param_info in enumerate(time_scaling_params):
            current_row = row + 1 + i
            
            if len(param_info) == 4:
                label, key, default, param_type = param_info
            else:
                label, key, default = param_info
                param_type = 'float'
            
            ttk.Label(frame, text=label).grid(row=current_row, column=0, sticky='w', padx=5, pady=5)
            
            if param_type == 'bool':
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(frame, variable=var)
            elif param_type == 'int':
                var = tk.IntVar(value=default)
                widget = ttk.Entry(frame, textvariable=var, width=15)
            else:  # float
                var = tk.DoubleVar(value=default)
                widget = ttk.Entry(frame, textvariable=var, width=15)
            
            widget.grid(row=current_row, column=1, padx=5, pady=5)
            self.param_vars[key] = var
            
            # Add help text for time scale factor
            if key == "time_scale_factor":
                help_text = ttk.Label(frame, text="(>1.0 = faster, e.g., 1000 for 1000x speed)", 
                                    font=('TkDefaultFont', 8, 'italic'))
                help_text.grid(row=current_row, column=2, sticky='w', padx=5)

        # ADAPTIVE MESH REFINEMENT SECTION
        row = row + len(time_scaling_params) + 1
        
        # Add separator
        separator2 = ttk.Separator(frame, orient='horizontal')
        separator2.grid(row=row, column=0, columnspan=3, sticky='ew', padx=5, pady=10)
        
        row += 1
        ttk.Label(frame, text="Adaptive Mesh Refinement", 
                font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, 
                                                        sticky='w', padx=5, pady=5)

        adaptive_params = [
            ("Enable Adaptive Refinement", "adaptive_refinement", True, 'bool'),
            ("Refinement Interval (steps)", "refinement_interval", 10, 'int'),
            ("Refinement Threshold", "refinement_threshold", 0.6, 'float'),
            ("Max Refinement Levels", "max_refinement_levels", 3, 'int'),
            ("Min Cell Size (m)", "min_cell_size", 1e-5, 'float'),
        ]
        
        for i, param_info in enumerate(adaptive_params):
            current_row = row + 1 + i
            
            if len(param_info) == 4:
                label, key, default, param_type = param_info
            else:
                label, key, default = param_info
                param_type = 'float'
            
            ttk.Label(frame, text=label).grid(row=current_row, column=0, sticky='w', padx=5, pady=5)
            
            if param_type == 'bool':
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(frame, variable=var)
            elif param_type == 'int':
                var = tk.IntVar(value=default)
                widget = ttk.Entry(frame, textvariable=var, width=15)
            else:  # float
                var = tk.DoubleVar(value=default)
                widget = ttk.Entry(frame, textvariable=var, width=15)
            
            widget.grid(row=current_row, column=1, padx=5, pady=5)
            self.param_vars[key] = var

        # Add time scaling info display at the bottom
        info_frame = ttk.LabelFrame(frame, text="Time Scaling Info")
        info_frame.grid(row=current_row+2, column=0, columnspan=3, sticky='ew', padx=5, pady=10)
        
        self.time_info_label = ttk.Label(info_frame, text="No time scaling", 
                                    font=('TkDefaultFont', 9), foreground="gray")
        self.time_info_label.pack(padx=10, pady=5)
        
        # Bind events to update info
        for key in ['scale_thermal_properties', 'time_scale_factor', 'total_time']:
            if key in self.param_vars:
                if isinstance(self.param_vars[key], tk.BooleanVar):
                    # For checkboxes, we need to trace the variable
                    self.param_vars[key].trace('w', lambda *args: self.update_time_scaling_info())

    def update_time_scaling_info(self):
        """Update time scaling information display"""
        try:
            if hasattr(self, 'time_info_label'):
                scale_enabled = self.param_vars.get('scale_thermal_properties', tk.BooleanVar()).get()
                scale_factor = self.param_vars.get('time_scale_factor', tk.DoubleVar(value=1.0)).get()
                total_time = self.param_vars.get('total_time', tk.DoubleVar(value=1.0)).get()
                
                if scale_enabled and scale_factor > 1.0:
                    scaled_time = total_time / scale_factor
                    info_text = (f"Time Scaling Active:\n"
                            f"Real time: {total_time:.3f} s\n"
                            f"Computational time: {scaled_time:.6f} s\n"
                            f"Speedup: {scale_factor}x")
                    self.time_info_label.config(text=info_text, foreground="green")
                else:
                    self.time_info_label.config(text="No time scaling", foreground="gray")
        except:
            pass  # Ignore errors during initialization

    def load_and_analyze_power_file(self, filename):
        """Load power file and extract time parameters"""
        try:
            import pandas as pd
            
            # Read the file
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)
            
            if df.shape[1] < 2:
                raise ValueError("Power file must have at least 2 columns: time and power percentage")
            
            # Get time data (first column)
            times = df.iloc[:, 0].values
            
            if len(times) < 2:
                raise ValueError("Power file must have at least 2 time points")
            
            # Sort times to ensure proper order
            times = np.sort(times)
            
            # Calculate parameters
            total_time = float(times[-1])  # Last time point
            
            # Calculate average time step
            time_diffs = np.diff(times)
            avg_dt = float(np.mean(time_diffs))
            min_dt = float(np.min(time_diffs))
            
            # Use minimum time step for better accuracy, but cap it for performance
            suggested_dt = max(min_dt, avg_dt / 10.0)  # At least 10x smaller than average
            suggested_dt = min(suggested_dt, 0.001)     # Cap at 0.01s for performance
            
            return {
                'total_time': total_time,
                'dt': suggested_dt,
                'file_points': len(times),
                'time_range': (float(times[0]), float(times[-1]))
            }
            
        except Exception as e:
            raise ValueError(f"Error analyzing power file: {str(e)}")

    def update_boundary_tab(self):
        """Update boundary tab based on simulation type"""
        # Clear existing widgets
        for widget in self.boundary_frame.winfo_children():
            widget.destroy()
            
        sim_type = self.sim_type_var.get()
        
        if sim_type == "fixed_temp":
            # Fixed temperature parameters
            params = [
                ("Fixed Temperature (K)", "fixed_temp", 800.0),
                ("Boundary Location", "boundary_location", "left"),
            ]
            
            for i, (label, key, default) in enumerate(params):
                ttk.Label(self.boundary_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
                if key == "boundary_location":
                    var = tk.StringVar(value=default)
                    combo = ttk.Combobox(self.boundary_frame, textvariable=var,
                                       values=["left", "right", "top", "bottom"], state="readonly")
                    combo.grid(row=i, column=1, padx=5, pady=5)
                else:
                    var = tk.DoubleVar(value=default)
                    entry = ttk.Entry(self.boundary_frame, textvariable=var, width=15)
                    entry.grid(row=i, column=1, padx=5, pady=5)
                self.param_vars[key] = var
                
        elif sim_type == "laser_heating":
            # Laser heating parameters
            params = [
                ("Peak Laser Power (W)", "peak_laser_power", 300.0),
                ("Beam Radius (m)", "beam_radius", 0.0005),
                ("Absorptivity", "absorptivity", 0.9),
            ]
            
            for i, (label, key, default) in enumerate(params):
                ttk.Label(self.boundary_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
                var = tk.DoubleVar(value=default)
                entry = ttk.Entry(self.boundary_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, padx=5, pady=5)
                self.param_vars[key] = var
            
            # Add power profile option
            row = len(params)
            ttk.Label(self.boundary_frame, text="Power Profile").grid(row=row, column=0, sticky='w', padx=5, pady=5)
            power_profile_var = tk.StringVar(value="constant")
            power_combo = ttk.Combobox(self.boundary_frame, textvariable=power_profile_var,
                                    values=["constant", "from_file"], state="readonly", width=12)
            power_combo.grid(row=row, column=1, padx=5, pady=5)
            self.param_vars['power_profile'] = power_profile_var
            
            # Add file selection
            row += 1
            ttk.Label(self.boundary_frame, text="Power File").grid(row=row, column=0, sticky='w', padx=5, pady=5)
            file_frame = ttk.Frame(self.boundary_frame)
            file_frame.grid(row=row, column=1, padx=5, pady=5, sticky='w')
            
            self.power_file_var = tk.StringVar(value="")
            file_entry = ttk.Entry(file_frame, textvariable=self.power_file_var, width=20, state='readonly')
            file_entry.pack(side='left')
            
            def browse_power_file():
                filename = filedialog.askopenfilename(
                    title="Select Power Profile File",
                    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                if filename:
                    try:
                        # Analyze the power file
                        file_info = self.load_and_analyze_power_file(filename)
                        
                        # Set the filename
                        self.power_file_var.set(filename)
                        
                        # Ask user if they want to update simulation parameters
                        msg = (f"Power file loaded successfully!\n\n"
                            f"File contains {file_info['file_points']} time points\n"
                            f"Time range: {file_info['time_range'][0]:.3f} - {file_info['time_range'][1]:.3f} s\n"
                            f"Suggested total time: {file_info['total_time']:.3f} s\n"
                            f"Suggested time step: {file_info['dt']:.4f} s\n\n"
                            f"Update simulation parameters automatically?")
                        
                        if messagebox.askyesno("Update Parameters?", msg):
                            # Update the simulation parameters
                            if 'total_time' in self.param_vars:
                                self.param_vars['total_time'].set(file_info['total_time'])
                            if 'dt' in self.param_vars:
                                self.param_vars['dt'].set(file_info['dt'])
                            
                            # Calculate and update output interval for reasonable number of outputs
                            target_outputs = 100  # Target ~100 output points
                            suggested_interval = max(1, int(file_info['total_time'] / file_info['dt'] / target_outputs))
                            if 'output_interval' in self.param_vars:
                                self.param_vars['output_interval'].set(suggested_interval)
                            
                            messagebox.showinfo("Parameters Updated", 
                                f"Simulation parameters updated:\n"
                                f"• Total time: {file_info['total_time']:.3f} s\n"
                                f"• Time step: {file_info['dt']:.4f} s\n"
                                f"• Output interval: {suggested_interval} steps")
                        
                    except Exception as e:
                        messagebox.showerror("File Error", f"Error loading power file:\n{str(e)}")
                        self.power_file_var.set("")  # Clear the filename on error
            
            browse_btn = ttk.Button(file_frame, text="Browse", command=browse_power_file)
            browse_btn.pack(side='left', padx=(5, 0))
            
            self.param_vars['power_file'] = self.power_file_var

    def create_control_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.run_button = ttk.Button(button_frame, text="Run Simulation", 
                                   command=self.run_simulation)
        self.run_button.pack(side='left', padx=5)
            
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                    command=self.stop_simulation, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Load Config", 
                  command=self.load_config).pack(side='left', padx=5)

    def create_status_panel(self, parent):
        """Create status and progress panel"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Status label
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(anchor='w')
        
         # Timer display
        timer_frame = ttk.Frame(status_frame)
        timer_frame.pack(fill='x', pady=2)
            
        ttk.Label(timer_frame, text="Elapsed Time:").pack(side='left')
        timer_label = ttk.Label(timer_frame, textvariable=self.timer_var, 
                            font=('Courier', 12, 'bold'), foreground="green")
        timer_label.pack(side='left', padx=(10, 0))

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.pack(fill='x', pady=2)

    def create_plot_area(self, parent):
        """Create the plotting area"""
        # Create matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Simulation Results")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize empty plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_title("No data")
            ax.grid(True, alpha=0.3)

    def on_sim_type_change(self):
        """Handle simulation type change"""
        self.update_boundary_tab()
        self.clear_results()

    def get_parameters(self):
        """Get current parameters from GUI"""
        params = {}
        for key, var in self.param_vars.items():
            try:
                params[key] = var.get()
            except tk.TclError:
                # Handle empty or invalid values
                params[key] = 0.0 if isinstance(var, (tk.DoubleVar, tk.IntVar)) else ""
        if hasattr(self, 'power_file_var'):
            params['power_file'] = self.power_file_var.get()

        return params

    def validate_current_parameters(self):
        """Validate current parameters"""
        params = self.get_parameters()
        errors = []
        
        # Basic validation
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if key in ['k', 'rho', 'cp', 'length', 'width', 'height', 'total_time', 'dt'] and value <= 0:
                    errors.append(f"{key} must be positive")
                elif key == 'mesh_resolution' and value < 4:
                    errors.append("Mesh resolution must be at least 4")
        
        # Simulation-specific validation
        sim_type = self.sim_type_var.get()
        if sim_type == "fixed_temp":
            if 'fixed_temp' not in params or params['fixed_temp'] <= 0:
                errors.append("Fixed temperature must be positive")
        elif sim_type == "laser_heating":
            if 'peak_laser_power' not in params or params['peak_laser_power'] <= 0:
                errors.append("Laser power must be positive")
            # Validate power file if using from_file profile
            if params.get('power_profile') == 'from_file':
                power_file = params.get('power_file', '')
                if not power_file:
                    errors.append("Power file is required when using 'from_file' profile")
                elif not Path(power_file).exists():
                    errors.append("Power file does not exist")

        # Time scaling validation
        if params.get('scale_thermal_properties', False):
            scale_factor = params.get('time_scale_factor', 1.0)
            if scale_factor <= 0:
                errors.append("Time scale factor must be positive")
            elif scale_factor > 10000:
                errors.append("Time scale factor too large (max 10000x recommended)")
            
            # Check if time scaling makes sense
            total_time = params.get('total_time', 1.0)
            dt = params.get('dt', 0.001)
            
            if scale_factor > 1.0:
                scaled_time = total_time / scale_factor
                scaled_dt = dt / scale_factor
                
                if scaled_time < 0.001:
                    errors.append(f"Scaled total time ({scaled_time:.6f}s) too small")
                if scaled_dt < 1e-9:
                    errors.append(f"Scaled time step ({scaled_dt:.2e}s) too small")
        
        # Emissivity validation
        emissivity = params.get('emissivity', 0.8)
        if not (0.0 <= emissivity <= 1.0):
            errors.append("Emissivity must be between 0.0 and 1.0")
            
        return errors

    def run_simulation(self):
        """Main simulation runner - called from GUI thread"""
        if self.simulation_running:
            return
    
        # Validate parameters first
        errors = self.validate_current_parameters()
        if errors:
            messagebox.showerror("Parameter Error", "\n".join(errors))
            return
        
        # Update UI state
        self.simulation_running = True
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("Starting simulation...")
        self.progress_var.set(0)
        
        # Start timer
        self.start_timer()

        # Clear previous results
        self.results = None
        self.clear_plots()
        
        # Start background thread
        self.simulation_thread = threading.Thread(target=self._simulation_worker)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def _simulation_worker(self):
        """Background worker - only communicates via root.after()"""
        try:
            # Get parameters (thread-safe copy)
            params = self.get_parameters()
            
            # Parameter name mapping for compatibility
            if 'laser_power' in params and 'peak_laser_power' not in params:
                params['peak_laser_power'] = params['laser_power']
            
            sim_type = self.sim_type_var.get()
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Creating simulation..."))
            
            # Create simulation
            if sim_type == "fixed_temp":
                self.current_simulation = FixedTempSimulation()
            elif sim_type == "laser_heating":
                self.current_simulation = LaserHeatingSimulation()
            else:
                raise ValueError(f"Unknown simulation type: {sim_type}")
            
            # Set parameters
            self.current_simulation.set_parameters(params)
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Running simulation..."))
            
            # Run simulation with progress callback
            self.results = self.current_simulation.run(
                progress_callback=self._thread_safe_progress_update
            )       

            # Success - update UI via main thread
            self.root.after(0, self._simulation_success)
            
        except Exception as e:
            # Error - update UI via main thread
            error_msg = str(e)
            print(f"Simulation error: {error_msg}")  # Debug output
            self.root.after(0, lambda: self._simulation_error(error_msg))

    def _thread_safe_progress_update(self, progress):
        """Thread-safe progress update"""
        self.root.after(0, lambda: self.progress_var.set(progress))

    def _simulation_success(self):
        """Handle successful simulation completion - called from main thread"""
        self.simulation_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Stop timer but keep final time displayed
        self.stop_timer()
        
        # Calculate final time
        if self.start_time is not None:
            import time
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            final_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.status_var.set(f"Simulation completed in {final_time}")
        else:
            self.status_var.set("Simulation completed")
        
        self.progress_var.set(100)
        
        # Plot results
        if self.results:
            self.update_plots()

    def _simulation_error(self, error_msg):
        """Handle simulation error - called from main thread"""
        self.simulation_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Stop timer and show error
        self.stop_timer()
        if self.start_time is not None:
            import time
            total_time = time.time() - self.start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            self.status_var.set(f"Simulation failed after {minutes:02d}:{seconds:02d}")
        else:
            self.status_var.set("Simulation failed")
        
        messagebox.showerror("Simulation Error", f"Simulation failed:\n{error_msg}")

    def stop_simulation(self):
        """Stop the running simulation"""
        if not self.simulation_running:
            return
        
        # Request stop
        if self.current_simulation:
            self.current_simulation.stop()
        
        # Stop timer
        self.stop_timer()
        if self.start_time is not None:
            import time
            total_time = time.time() - self.start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            self.status_var.set(f"Simulation stopped after {minutes:02d}:{seconds:02d}")
        else:
            self.status_var.set("Simulation stopped")
        
        # Update UI
        self.simulation_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')

    # ==================== PLOT MANAGEMENT ====================

    def update_plots(self):
        """Update plots with simulation results"""
        if not self.results:
            return
            
        # Clear previous plots
        self.clear_plots()
        
        try:
            # Update plots based on results
            if self.current_simulation:
                self.current_simulation.plot_results(self.results, [self.ax1, self.ax2, self.ax3, self.ax4])
            
            plt.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            messagebox.showerror("Plot Error", f"Error updating plots: {e}")

    def clear_plots(self):
        """Clear all plots"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_title("No data")
            ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def clear_results(self):
        """Clear all results"""
        self.results = None
        self.clear_plots()
        self.status_var.set("Ready")
        self.progress_var.set(0)

    # ==================== FILE OPERATIONS ====================

    def save_results(self):
        """Save simulation results"""
        if not self.results:
            messagebox.showwarning("No Results", "No simulation results to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if self.current_simulation:
                    self.current_simulation.save_results(self.results, filename)
                    messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving results: {e}")

    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                if 'parameters' in config:
                    # Update GUI parameters
                    for key, value in config['parameters'].items():
                        if key in self.param_vars:
                            self.param_vars[key].set(value)
                    
                    messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                    
            except Exception as e:
                messagebox.showerror("Load Error", f"Error loading configuration: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalSimulationGUI(root)
    root.mainloop()