import os
import sys
import subprocess
import argparse

def main(args):
    """Run the Streamlit dashboard to visualize benchmarking results."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    # Determine results directory
    results_dir = os.path.abspath(args.results_dir)
    os.environ["BENCHMARK_RESULTS_DIR"] = results_dir
    
    # Check if any results exist
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory {results_dir} does not exist.")
        if not args.force:
            print("No benchmark results found. Please run the benchmarking script first.")
            print("If you want to start the dashboard anyway, use the --force flag.")
            return
    
    # Build the command
    cmd = [
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
    ]
    
    if args.browser:
        cmd.extend(["--server.headless", "false"])
    else:
        cmd.extend(["--server.headless", "true"])
    
    # Print information
    print(f"Starting dashboard to visualize benchmarking results from: {results_dir}")
    print(f"Dashboard will be available at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the dashboard")
    
    # Run the dashboard
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scCausalVI vs Model1 benchmarking dashboard")
    parser.add_argument("--results-dir", type=str, default=os.path.join("data", "benchmarking_results"),
                        help="Directory containing benchmark results")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the dashboard on")
    parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
    parser.add_argument("--force", action="store_true", help="Start the dashboard even if no results exist")
    
    args = parser.parse_args()
    main(args)
