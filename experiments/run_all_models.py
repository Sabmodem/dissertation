#!/usr/bin/env python3
"""
Script to run all fraud detection models and compare results
"""
import os
import sys
import subprocess
import time
from datetime import datetime

# Force CPU for TensorFlow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_model(script_name, timeout=300):
    """Run a model script and return success status"""
    print(f"Running {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully in {elapsed_time:.2f}s")
            print("\nOutput:")
            # Print only lines with metrics
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in 
                      ['accuracy', 'precision', 'recall', 'f1', 'results', '===']):
                    print(line)
            return True
        else:
            print(f"❌ {script_name} failed")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {script_name} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {script_name} error: {str(e)}")
        return False

def main():
    print_header(f"Fraud Detection Models Test Suite - {datetime.now()}")
    
    # List of models to test (script_name, timeout_seconds)
    models = [
        ("Logistic_Regression.py", 120),
        ("ANN_fixed.py", 300),
        ("XGBoost_fixed.py", 120),
        ("RandomForest.py", 180),
        ("SVM.py", 180),
    ]
    
    results = {}
    
    for script, timeout in models:
        print_header(f"Testing: {script}")
        if os.path.exists(script):
            results[script] = run_model(script, timeout)
        else:
            print(f"⚠️ {script} not found, skipping...")
            results[script] = None
        print()
    
    # Summary
    print_header("Test Summary")
    for script, status in results.items():
        status_symbol = "✅" if status else ("❌" if status is False else "⚠️")
        status_text = "PASSED" if status else ("FAILED" if status is False else "SKIPPED")
        print(f"{status_symbol} {script:40s} - {status_text}")
    
    # Count results
    passed = sum(1 for s in results.values() if s is True)
    failed = sum(1 for s in results.values() if s is False)
    skipped = sum(1 for s in results.values() if s is None)
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
