#!/usr/bin/env python3
"""
MATLAB Workspace Viewer for Cursor
Connects to MATLAB Engine and provides workspace variable inspection
"""

import matlab.engine
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import json

class MATLABWorkspaceViewer:
    def __init__(self):
        """Initialize MATLAB Engine connection"""
        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
        print("✅ Connected to MATLAB!")
        
    def get_workspace_variables(self) -> Dict[str, Any]:
        """Get all variables from MATLAB workspace"""
        try:
            # Get workspace info
            workspace_info = self.eng.workspace
            variables = {}
            
            # Get variable names
            var_names = self.eng.evalc('who')
            var_list = [name.strip() for name in var_names.strip().split('\n') if name.strip()]
            
            for var_name in var_list:
                try:
                    # Get variable value
                    var_value = workspace_info[var_name]
                    variables[var_name] = {
                        'value': var_value,
                        'type': str(type(var_value)),
                        'size': self._get_size(var_value)
                    }
                except Exception as e:
                    variables[var_name] = {
                        'error': f"Could not retrieve: {str(e)}",
                        'type': 'error'
                    }
            
            return variables
            
        except Exception as e:
            print(f"Error getting workspace variables: {e}")
            return {}
    
    def _get_size(self, obj) -> str:
        """Get size information for MATLAB objects"""
        try:
            if hasattr(obj, '__len__'):
                if hasattr(obj, 'shape'):
                    return str(obj.shape)
                else:
                    return str(len(obj))
            else:
                return "scalar"
        except:
            return "unknown"
    
    def get_variable_details(self, var_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific variable"""
        try:
            workspace_info = self.eng.workspace
            var_value = workspace_info[var_name]
            
            details = {
                'name': var_name,
                'type': str(type(var_value)),
                'size': self._get_size(var_value)
            }
            
            # Convert to Python types for better display
            if hasattr(var_value, '_data'):
                # MATLAB array
                details['data'] = np.array(var_value._data)
                details['shape'] = details['data'].shape
                details['dtype'] = str(details['data'].dtype)
                
                # Show sample data
                if details['data'].size <= 100:
                    details['sample'] = details['data'].tolist()
                else:
                    details['sample'] = details['data'].flatten()[:10].tolist()
                    details['note'] = f"Showing first 10 of {details['data'].size} elements"
                    
            elif isinstance(var_value, (int, float, str, bool)):
                details['value'] = var_value
                
            return details
            
        except Exception as e:
            return {'error': f"Could not get details for {var_name}: {str(e)}"}
    
    def run_matlab_command(self, command: str) -> str:
        """Run a MATLAB command and return the result"""
        try:
            result = self.eng.evalc(command)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def close(self):
        """Close MATLAB Engine connection"""
        if hasattr(self, 'eng'):
            self.eng.quit()
            print("MATLAB Engine closed.")

def main():
    """Main function to demonstrate usage"""
    viewer = MATLABWorkspaceViewer()
    
    try:
        # Example: Run some MATLAB code
        print("\n=== Running MATLAB Code ===")
        viewer.run_matlab_command("a = 1:10;")
        viewer.run_matlab_command("b = rand(5,5);")
        viewer.run_matlab_command("c = 'Hello from MATLAB';")
        
        # Get workspace variables
        print("\n=== Workspace Variables ===")
        variables = viewer.get_workspace_variables()
        
        for name, info in variables.items():
            size_info = info.get('size', 'unknown')
            type_info = info.get('type', 'unknown')
            print(f"{name}: {type_info} - Size: {size_info}")
        
        # Get detailed info for a specific variable
        print("\n=== Variable Details ===")
        if 'b' in variables:
            details = viewer.get_variable_details('b')
            print(f"Variable 'b' details:")
            print(json.dumps(details, indent=2, default=str))
        
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("Type MATLAB commands (or 'quit' to exit):")
        
        while True:
            try:
                command = input("MATLAB> ")
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if command.strip():
                    result = viewer.run_matlab_command(command)
                    print(result)
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    finally:
        viewer.close()

if __name__ == "__main__":
    main() 