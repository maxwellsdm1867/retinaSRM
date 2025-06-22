#!/usr/bin/env python3
"""
MATLAB Workspace Viewer for Cursor
Run this script directly in Cursor to view MATLAB workspace variables
"""

import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting MATLAB Workspace Viewer for Cursor...")
    
    # Step 1: Connect to MATLAB
    print("\n1️⃣ Connecting to MATLAB Engine...")
    try:
        eng = matlab.engine.start_matlab()
        print("✅ Connected to MATLAB!")
    except Exception as e:
        print(f"❌ Failed to connect to MATLAB: {e}")
        return
    
    # Step 2: Create workspace viewer class
    class MATLABWorkspaceViewer:
        def __init__(self, engine):
            self.eng = engine
            
        def get_workspace_variables(self) -> Dict[str, Any]:
            """Get all variables from MATLAB workspace"""
            try:
                workspace_info = self.eng.workspace
                variables = {}
                
                # Get variable names
                var_names = self.eng.evalc('who')
                var_list = [name.strip() for name in var_names.strip().split('\n') if name.strip()]
                
                for var_name in var_list:
                    try:
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
        
        def display_workspace_table(self):
            """Display workspace variables in a nice table"""
            variables = self.get_workspace_variables()
            
            if not variables:
                print("No variables found in workspace.")
                return
            
            print(f"\n📊 MATLAB Workspace Variables ({len(variables)} variables):")
            print("=" * 60)
            print(f"{'Variable':<15} {'Type':<25} {'Size':<15}")
            print("-" * 60)
            
            for name, info in variables.items():
                var_type = info.get('type', 'unknown')
                var_size = info.get('size', 'unknown')
                print(f"{name:<15} {var_type:<25} {var_size:<15}")
    
    # Create viewer instance
    viewer = MATLABWorkspaceViewer(eng)
    print("✅ Workspace viewer created!")
    
    # Step 3: Create sample data
    print("\n2️⃣ Creating sample MATLAB variables...")
    viewer.run_matlab_command("a = 1:100;")
    viewer.run_matlab_command("b = rand(20, 20);")
    viewer.run_matlab_command("c = sin(linspace(0, 4*pi, 1000));")
    viewer.run_matlab_command("d = 'Hello from MATLAB';")
    viewer.run_matlab_command("e = [1, 2, 3; 4, 5, 6; 7, 8, 9];")
    print("✅ Sample variables created!")
    
    # Step 4: Display workspace variables
    print("\n3️⃣ Displaying workspace variables...")
    viewer.display_workspace_table()
    
    # Step 5: Show detailed info for a specific variable
    print("\n4️⃣ Detailed information for variable 'b':")
    details = viewer.get_variable_details('b')
    print(json.dumps(details, indent=2, default=str))
    
    # Step 6: Change to your project directory
    print("\n5️⃣ Changing to project directory...")
    viewer.run_matlab_command("cd('/Users/maxwellsdm/Documents/GitHub/matlabPyrTools/retinaSRM');")
    print("✅ Changed to project directory!")
    
    # Step 7: Interactive mode
    print("\n6️⃣ Interactive MATLAB Commands")
    print("=" * 40)
    print("You can now run MATLAB commands interactively.")
    print("Type 'quit' to exit, or run commands like:")
    print("  - whos (list all variables)")
    print("  - run('curreinjt_ai_population.m') (run your script)")
    print("  - a = 1:10; (create variables)")
    print("  - plot(1:100, sin(1:100)) (create plots)")
    
    while True:
        try:
            command = input("\nMATLAB> ")
            if command.lower() in ['quit', 'exit', 'q']:
                break
                
            if command.strip():
                result = viewer.run_matlab_command(command)
                print(result)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break
    
    # Clean up
    print("\n7️⃣ Cleaning up...")
    eng.quit()
    print("✅ MATLAB Engine closed.")
    print("🎉 Done! You can now view MATLAB workspace variables in Cursor!")

if __name__ == "__main__":
    main() 