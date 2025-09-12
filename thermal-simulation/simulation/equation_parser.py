# equation_parser.py
"""
Safe equation parser for temperature-dependent material properties.
Supports mathematical expressions with temperature as a variable.
"""

import ast
import operator
import math
import numpy as np


class EquationParser:
    """Safe equation parser for temperature-dependent properties"""
    
    # Allowed operations and functions
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    ALLOWED_FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'abs': abs,
        'min': min,
        'max': max,
        'pow': pow,
        # NumPy functions for array operations
        'np_sin': np.sin,
        'np_cos': np.cos,
        'np_tan': np.tan,
        'np_exp': np.exp,
        'np_log': np.log,
        'np_sqrt': np.sqrt,
        'np_abs': np.abs,
    }
    
    ALLOWED_CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self, equation_str, variables):
        """
        Initialize parser with equation string and available variables
        
        Args:
            equation_str: String equation like "k_base * (1 + 0.001 * T)"
            variables: Dict of variable names and values like {'T': 500, 'k_base': 120}
        """
        self.equation_str = equation_str
        self.variables = variables
        self.compiled_expr = None
        self._compile_equation()
    
    def _compile_equation(self):
        """Compile the equation for faster evaluation"""
        try:
            # Parse the equation
            tree = ast.parse(self.equation_str, mode='eval')
            self.compiled_expr = compile(tree, '<string>', 'eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid equation syntax: {self.equation_str}\nError: {e}")
    
    def evaluate(self, **kwargs):
        """Evaluate the equation with given variable values"""
        # Combine default variables with provided kwargs
        eval_vars = {**self.variables, **kwargs}
        
        # Add allowed functions and constants
        eval_vars.update(self.ALLOWED_FUNCTIONS)
        eval_vars.update(self.ALLOWED_CONSTANTS)
        
        try:
            # Evaluate the compiled expression
            result = eval(self.compiled_expr, {"__builtins__": {}}, eval_vars)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating equation '{self.equation_str}' with variables {kwargs}: {e}")
    
    def evaluate_array(self, T_array):
        """Evaluate equation for an array of temperatures"""
        if isinstance(T_array, (list, tuple)):
            T_array = np.array(T_array)
        
        # For array operations, we need to use numpy functions
        eval_vars = {**self.variables}
        eval_vars.update(self.ALLOWED_FUNCTIONS)
        eval_vars.update(self.ALLOWED_CONSTANTS)
        eval_vars['T'] = T_array
        
        # Replace math functions with numpy equivalents for array operations
        equation_np = self.equation_str
        replacements = {
            'sin(': 'np_sin(',
            'cos(': 'np_cos(',
            'tan(': 'np_tan(',
            'exp(': 'np_exp(',
            'log(': 'np_log(',
            'sqrt(': 'np_sqrt(',
            'abs(': 'np_abs(',
        }
        
        for old, new in replacements.items():
            equation_np = equation_np.replace(old, new)
        
        try:
            tree = ast.parse(equation_np, mode='eval')
            compiled_expr = compile(tree, '<string>', 'eval')
            result = eval(compiled_expr, {"__builtins__": {}}, eval_vars)
            return np.array(result, dtype=float)
        except Exception as e:
            raise ValueError(f"Error evaluating equation '{equation_np}' for array: {e}")

    @staticmethod
    def validate_equation(equation_str, test_variables):
        """Validate that an equation is safe and evaluates correctly"""
        try:
            parser = EquationParser(equation_str, test_variables)
            # Test with sample values
            result = parser.evaluate(T=500.0)
            if not isinstance(result, (int, float)) or not math.isfinite(result):
                raise ValueError("Equation must return a finite number")
            return True, None
        except Exception as e:
            return False, str(e)