#!/usr/bin/env python3

import time
import numpy as np
from agent.xgboost_convert.json_xgboost import JSON_XGBoost
from agent.xgboost_convert.numpy_xgboost import NP_XGBoost
from agent.constants import *
import os
import shutil

def benchmark_xgboost():
    """Benchmark the optimized JSON XGBoost implementation"""
    # Create a sample board for testing
    board = np.zeros((BOARD_N, BOARD_N), dtype=int)
    
    # Place some frogs and lilypads
    # Red frogs
    board[0, 0] = RED
    board[1, 1] = RED
    board[2, 2] = RED
    board[2, 4] = RED
    board[3, 3] = RED
    board[4, 6] = RED
    
    # Blue frogs
    board[7, 7] = BLUE
    board[6, 6] = BLUE
    board[5, 5] = BLUE
    board[5, 3] = BLUE
    board[4, 4] = BLUE
    board[3, 1] = BLUE
    
    # Some lilypads
    board[2, 3] = LILYPAD
    board[3, 2] = LILYPAD
    board[4, 5] = LILYPAD
    board[5, 4] = LILYPAD
    
    # First prediction with JSON_XGBoost (includes loading model)
    print("Initializing JSON_XGBoost model and making first prediction...")
    json_model = JSON_XGBoost()
    start_time = time.time()
    json_score = json_model.predict(board, RED)
    json_first_time = time.time() - start_time
    print(f"First prediction (including model setup): {json_first_time:.4f} seconds")
    print(f"Score: {json_score:.4f}")
    
    # First prediction with NP_XGBoost
    print("\nInitializing NP_XGBoost model and making first prediction...")
    np_model = NP_XGBoost()
    start_time = time.time()
    np_score = np_model.predict(board, RED)
    np_first_time = time.time() - start_time
    print(f"First prediction (including model setup): {np_first_time:.4f} seconds")
    print(f"Score: {np_score:.4f}")
    
    # Subsequent predictions for JSON_XGBoost (should be much faster)
    num_iterations = 100
    print(f"\nBenchmarking JSON_XGBoost for {num_iterations} predictions...")
    start_time = time.time()
    for _ in range(num_iterations):
        json_score = json_model.predict(board, RED)
    json_total_time = time.time() - start_time
    json_avg_time = json_total_time / num_iterations
    
    print(f"Total time for {num_iterations} predictions: {json_total_time:.4f} seconds")
    print(f"Average time per prediction: {json_avg_time*1000:.2f} milliseconds")
    
    # Subsequent predictions for NP_XGBoost
    print(f"\nBenchmarking NP_XGBoost for {num_iterations} predictions...")
    start_time = time.time()
    for _ in range(num_iterations):
        np_score = np_model.predict(board, RED)
    np_total_time = time.time() - start_time
    np_avg_time = np_total_time / num_iterations
    
    print(f"Total time for {num_iterations} predictions: {np_total_time:.4f} seconds")
    print(f"Average time per prediction: {np_avg_time*1000:.2f} milliseconds")
    
    # Compare JSON vs NumPy implementations
    if np_avg_time < json_avg_time:
        print(f"\nNP_XGBoost is {json_avg_time/np_avg_time:.1f}x faster than JSON_XGBoost")
    else:
        print(f"\nJSON_XGBoost is {np_avg_time/json_avg_time:.1f}x faster than NP_XGBoost")
    
    # Compare with the original implementation if backed up
    backup_path = os.path.join(os.path.dirname(__file__), 'agent/xgboost_convert', 'json_xgboost_original.py')
    json_xgboost_path = os.path.join(os.path.dirname(__file__), 'agent/xgboost_convert', 'json_xgboost.py')
    
    try:
        # Check if this is the first run (before optimization)
        if not os.path.exists(backup_path):
            # Make a backup of the original implementation
            shutil.copy2(json_xgboost_path, backup_path)
            print(f"\nBackup of the original implementation saved to {backup_path}")
            print("Please run the optimization code and then run this benchmark again to compare.")
        else:
            # Check if we're running the optimized version
            with open(json_xgboost_path, 'r') as f:
                current_code = f.read()
            with open(backup_path, 'r') as f:
                original_code = f.read()
                
            if current_code == original_code:
                print("\nYou're running the benchmark on the original implementation.")
                print("Please apply your optimizations to json_xgboost.py and run this benchmark again.")
            else:
                # Import the original implementation
                import importlib.util
                spec = importlib.util.spec_from_file_location("json_xgboost_original", backup_path)
                original_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(original_module)
                
                OriginalXGBoost = original_module.SLOW_JSON_XGBoost
                
                print("\nComparing with original JSON_XGBoost implementation...")
                original_model = OriginalXGBoost()
                start_time = time.time()
                for _ in range(10):  # Fewer iterations for slower implementation
                    original_score = original_model.predict(board, RED)
                original_time = (time.time() - start_time) / 10
                print(f"Original implementation average time: {original_time*1000:.2f} milliseconds")
                print(f"Optimized JSON_XGBoost speed improvement: {original_time/json_avg_time:.1f}x faster")
                print(f"NP_XGBoost vs original: {original_time/np_avg_time:.1f}x faster")
                
                # Create a simple comparison table
                print("\n=== Performance Comparison ===")
                print(f"{'Implementation':<25} {'Time (ms)':<15} {'Speedup':<10}")
                print("-" * 50)
                print(f"{'Original JSON_XGBoost':<25} {original_time*1000:.2f}{'ms':<10} {1.0:.1f}x")
                print(f"{'Optimized JSON_XGBoost':<25} {json_avg_time*1000:.2f}{'ms':<10} {original_time/json_avg_time:.1f}x")
                print(f"{'NumPy XGBoost':<25} {np_avg_time*1000:.2f}{'ms':<10} {original_time/np_avg_time:.1f}x")
    except Exception as e:
        print(f"\nCouldn't compare implementations: {e}")

if __name__ == "__main__":
    benchmark_xgboost()