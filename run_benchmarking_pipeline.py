#!/usr/bin/env python3
"""
Main script to run the benchmarking pipeline:
1. Download datasets (if not already downloaded)
2. Run benchmarking on all datasets
3. Launch the Streamlit dashboard
"""
import os
import argparse
import subprocess
from pathlib import Path

def main(args):
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(project_root, args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create results directory
    results_dir = os.path.join(data_dir, "benchmarking_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Install requirements if needed
    if args.install_requirements:
        print("Installing requirements...")
        subprocess.run(["pip", "install", "-r", os.path.join(project_root, "requirements.txt")])
    
    # Step 1b: Download datasets if needed
    if args.download_datasets:
        print("Downloading datasets...")
        download_cmd = [
            "python", "-m", "sccvi_impl.data.Download_Dataset"
        ]
        if args.dataset:
            # If a specific dataset is specified, only download that one
            download_cmd.extend(["--dataset", args.dataset, "--output_dir", data_dir])
        else:
            # Otherwise download all datasets
            download_cmd.extend(["--output_dir", data_dir])
        
        subprocess.run(download_cmd)
    
    # Step 2: Run the benchmarking
    if args.run_benchmark:
        print("Running benchmarking...")
        benchmark_cmd = [
            "python", "-m", "sccvi_impl.benchmarking.run_benchmark",
            "--output_dir", data_dir,
            "--max_epochs", str(args.max_epochs)
        ]
        
        if args.dataset:
            benchmark_cmd.extend(["--dataset", args.dataset])
            
        subprocess.run(benchmark_cmd)
    
    # Step 3: Launch the dashboard
    if args.run_dashboard:
        print("Launching dashboard...")
        dashboard_cmd = [
            "python", "-m", "sccvi_impl.benchmarking.run_dashboard",
            "--results-dir", results_dir,
            "--port", str(args.port)
        ]
        
        if args.browser:
            dashboard_cmd.append("--browser")
            
        subprocess.run(dashboard_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the benchmarking pipeline")
    
    # General options
    parser.add_argument("--data_dir", type=str, default="data", 
                      help="Directory to store datasets and results")
    parser.add_argument("--install_requirements", action="store_true",
                      help="Install required packages before running")
    parser.add_argument("--download_datasets", action="store_true",
                      help="Download datasets before benchmarking")
    
    # Benchmarking options
    parser.add_argument("--run_benchmark", action="store_true",
                      help="Run the benchmarking process")
    parser.add_argument("--dataset", type=str,
                      help="Specific dataset to benchmark (default: all)")
    parser.add_argument("--max_epochs", type=int, default=50,
                      help="Maximum number of training epochs")
    
    # Dashboard options
    parser.add_argument("--run_dashboard", action="store_true",
                      help="Launch the Streamlit dashboard")
    parser.add_argument("--port", type=int, default=8501,
                      help="Port to run the dashboard on")
    parser.add_argument("--browser", action="store_true",
                      help="Open dashboard in browser")
    
    args = parser.parse_args()
    
    # If no specific action is specified, run everything
    if not any([args.run_benchmark, args.run_dashboard, args.download_datasets]):
        args.run_benchmark = True
        args.run_dashboard = True
        args.download_datasets = True
    
    main(args)
