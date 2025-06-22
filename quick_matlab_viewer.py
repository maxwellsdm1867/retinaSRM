#!/usr/bin/env python3
"""
Quick MATLAB Workspace Viewer
Simple script to quickly view MATLAB workspace variables
"""

import matlab.engine
import numpy as np

def quick_view():
    print("🔍 Quick MATLAB Workspace Viewer")
    print("=" * 40)
    
    # Connect to MATLAB
    eng = matlab.engine.start_matlab()
    
    try:
        # Get workspace variables
        var_names = eng.evalc('who')
        var_list = [name.strip() for name in var_names.strip().split('\n') if name.strip()]
        
        if not var_list:
            print("No variables in workspace.")
            return
        
        print(f"Found {len(var_list)} variables:")
        print("-" * 40)
        
        for var_name in var_list:
            try:
                # Get variable info
                size_info = eng.evalc(f'size({var_name})')
                class_info = eng.evalc(f'class({var_name})')
                
                print(f"📊 {var_name}: {class_info.strip()} - Size: {size_info.strip()}")
                
            except Exception as e:
                print(f"❌ {var_name}: Error getting info")
    
    finally:
        eng.quit()

if __name__ == "__main__":
    quick_view() 