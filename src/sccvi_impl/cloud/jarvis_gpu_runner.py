#!/usr/bin/env python3
"""
JarvisLabs GPU Runner for scCausalVI Benchmarking

This script provides utilities for running scCausalVI benchmarking on 
JarvisLabs GPU instances. It handles configuration, environment setup,
and monitoring of GPU usage.

Usage:
    # Local execution:
    python jarvis_gpu_runner.py --script simulated --epochs 200 --batch_size 256
    
    # Remote execution via SSH:
    python jarvis_gpu_runner.py --remote --host jarvis.example.com --port 22 \
        --username user --key ~/.ssh/id_rsa --script simulated
        
    # Remote execution with direct SSH string:
    python jarvis_gpu_runner.py --remote --ssh_string "ssh -p 11014 root@sshc.jarvislabs.ai" \
        --script simulated --epochs 50 --batch_size 64
"""

import os
import sys
import argparse
import subprocess
import time
import paramiko
import getpass
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run scCausalVI benchmarking on JarvisLabs GPU')
    
    parser.add_argument('--script', type=str, default='simulated',
                        choices=['simulated', 'covid_epithelial', 'covid_pbmc', 'ifn_beta'],
                        help='Which benchmarking script to run')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor GPU usage during training')
    
    parser.add_argument('--monitor_interval', type=int, default=30,
                        help='Interval in seconds between GPU monitoring checks')
    
    # SSH options for remote execution
    parser.add_argument('--remote', action='store_true', default=False,
                        help='Run benchmark on a remote JarvisLabs instance via SSH')
    
    parser.add_argument('--host', type=str, default=None,
                        help='Hostname or IP address of the remote JarvisLabs instance')
    
    parser.add_argument('--port', type=int, default=22,
                        help='SSH port for the remote JarvisLabs instance (default: 22)')
    
    parser.add_argument('--username', type=str, default=None,
                        help='SSH username for the remote JarvisLabs instance')
    
    parser.add_argument('--key', type=str, default=None,
                        help='Path to SSH private key file')
    
    parser.add_argument('--password', action='store_true', default=False,
                        help='Use password authentication instead of key-based authentication')
    
    parser.add_argument('--ssh_string', type=str, default=None,
                        help='Pre-formatted SSH connection string (e.g., "ssh -p PORT USER@HOST")')
    
    return parser.parse_args()

def get_script_path(script_name):
    """Get the path to the benchmarking script based on name."""
    # Get the path to the current file
    current_dir = Path(__file__).parent.absolute()
    
    # Navigate to the scripts directory
    scripts_dir = current_dir.parent / 'scripts'
    
    # Map script names to their file paths
    script_map = {
        'simulated': scripts_dir / 'simulated_benchmarking.py',
        'covid_epithelial': scripts_dir / 'covid_epithelial_benchmarking.py',
        'covid_pbmc': scripts_dir / 'covid_pbmc_benchmarking.py',
        'ifn_beta': scripts_dir / 'ifn_beta_benchmarking.py'
    }
    
    return str(script_map[script_name])

def monitor_gpu_usage(interval=30):
    """Start a separate process to monitor GPU usage."""
    try:
        import threading
        import subprocess
        
        def check_gpu():
            while True:
                print("\n--- GPU Status ---")
                subprocess.call("nvidia-smi", shell=True)
                print("-----------------\n")
                time.sleep(interval)
        
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=check_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f"GPU monitoring started (every {interval} seconds)")
        return monitor_thread
    
    except Exception as e:
        print(f"Failed to start GPU monitoring: {e}")
        return None

def run_benchmark(script_path, epochs, batch_size, monitor=False, monitor_interval=30):
    """Run the benchmarking script with GPU support."""
    # Ensure the script exists
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Check for GPU availability
    try:
        gpu_check = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_gpu = gpu_check.returncode == 0
    except:
        has_gpu = False
    
    if not has_gpu:
        print("Warning: No GPU detected. Benchmarking will run on CPU.")
    else:
        print("GPU detected. Will use GPU for benchmarking.")
    
    # Start GPU monitoring if requested and available
    if monitor and has_gpu:
        monitor_thread = monitor_gpu_usage(monitor_interval)
    
    # Build the command
    cmd = [
        "python", script_path,
        "--n_epochs", str(epochs),
        "--batch_size", str(batch_size)
    ]
    
    if has_gpu:
        cmd.append("--use_gpu")
    
    # Current directory context
    script_dir = os.path.dirname(script_path)
    
    # Run the benchmark
    print(f"Running benchmark: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        process = subprocess.run(cmd, check=True)
        success = process.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with error: {e}")
        success = False
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print summary
    print(f"\nBenchmark {'completed successfully' if success else 'failed'}")
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    return success

def run_benchmark_remote(args):
    """Run the benchmarking script on a remote JarvisLabs instance via SSH."""
    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # If SSH string is provided, parse it
    if args.ssh_string:
        import re
        # Match patterns like: ssh -o StrictHostKeyChecking=no -p 11014 root@sshc.jarvislabs.ai
        ssh_pattern = re.compile(r'ssh\s+(?:-o\s+[\w=]+\s+)*(?:-p\s+(\d+))?\s+([\w-]+)@([\w.-]+)')
        match = ssh_pattern.match(args.ssh_string)
        
        if match:
            port_str, username, host = match.groups()
            port = int(port_str) if port_str else 22
            args.port = port
            args.username = username
            args.host = host
            print(f"Parsed SSH string: {username}@{host}:{port}")
        else:
            print(f"Error: Could not parse SSH string: {args.ssh_string}")
            return False
    
    if not args.host:
        print("Error: Host is required for remote execution. Use --host option or --ssh_string.")
        return False
    
    if not args.username:
        print("Error: Username is required for remote execution. Use --username option or --ssh_string.")
        return False
    
    temp_archive = None
    
    try:
        print(f"Connecting to {args.host}:{args.port} as {args.username}...")
        
        # Handle authentication
        if args.password or args.ssh_string:
            password = getpass.getpass("SSH Password: ")
            client.connect(args.host, port=args.port, username=args.username, password=password)
        else:
            key_path = args.key or os.path.expanduser("~/.ssh/id_rsa")
            if not os.path.exists(key_path):
                print(f"Error: SSH key not found at {key_path}")
                password = getpass.getpass("SSH Password (fallback): ")
                client.connect(args.host, port=args.port, username=args.username, password=password)
                return False
            
            try:
                key = paramiko.RSAKey.from_private_key_file(key_path)
                client.connect(args.host, port=args.port, username=args.username, pkey=key)
            except paramiko.ssh_exception.PasswordRequiredException:
                passphrase = getpass.getpass("SSH Key Passphrase: ")
                key = paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
                client.connect(args.host, port=args.port, username=args.username, pkey=key)
        
        # Check GPU availability on remote host
        print("Checking GPU availability on remote host...")
        _, stdout, stderr = client.exec_command("nvidia-smi")
        gpu_output = stdout.read().decode()
        gpu_error = stderr.read().decode()
        
        if gpu_error and "not found" in gpu_error:
            print("Warning: No GPU detected on remote host. Benchmarking will run on CPU.")
            has_gpu = False
        else:
            print("GPU detected on remote host:")
            print(gpu_output)
            has_gpu = True
        
        # Get the current directory structure on remote host
        print("Setting up remote environment...")
        _, stdout, _ = client.exec_command("echo $HOME")
        home_dir = stdout.read().decode().strip()
        
        # Remote paths
        remote_work_dir = f"{home_dir}/sccvi_workspace"
        
        # Create necessary directories
        print("Creating remote workspace directory...")
        client.exec_command(f"mkdir -p {remote_work_dir}/scripts")

        
        
        # Create a temporary directory for the project
        import tempfile
        import shutil
        
        # Get the project root directory (two levels up from the current file)
        project_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
        
        print(f"Project directory: {project_dir}")
        
        # Create SFTP connection
        sftp = client.open_sftp()
        
        # Create remote project directory
        remote_project_dir = f"{remote_work_dir}/sccvi_impl"
        client.exec_command(f"mkdir -p {remote_project_dir}")
        
        # Upload the entire project directory
        print("Uploading project files to remote server...")
        
        def upload_directory(local_dir, remote_dir):
            """Recursively upload a directory to remote server."""
            for item in os.listdir(local_dir):
                if item.startswith('.') or item == '__pycache__' or item.endswith('.pyc'):
                    continue
                
                local_path = os.path.join(local_dir, item)
                remote_path = f"{remote_dir}/{item}"
                
                if os.path.isfile(local_path):
                    print(f"Uploading {local_path} to {remote_path}")
                    sftp.put(local_path, remote_path)
                elif os.path.isdir(local_path):
                    # Create the remote directory
                    try:
                        sftp.mkdir(remote_path)
                    except IOError:
                        pass  # Directory might already exist
                    
                    # Upload the directory contents
                    upload_directory(local_path, remote_path)
        
        upload_directory(project_dir, remote_work_dir)
        
        # Get the script path (just for reference, we'll use the uploaded one)
        script_path = get_script_path(args.script)
        script_name = os.path.basename(script_path)
        
        # Install required packages and the project in development mode
        print("Installing required packages and project on remote server...")
        
        # First, explicitly install gdown which is needed for downloading datasets
        print("Installing gdown package...")
        _, stdout, stderr = client.exec_command("pip install gdown scikit-misc")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        
        if stderr_result and "error" in stderr_result.lower():
            print(f"Error installing gdown: {stderr_result}")
        else:
            print("gdown and scikit-misc installed successfully.")
        
        # Explicitly install all requirements from requirements.txt first
        print("Installing all requirements from requirements.txt...")
        req_install_cmd = f"cd {remote_work_dir}/sccvi_impl && pip install -r requirements.txt"
        print(f"Running requirements installation command: {req_install_cmd}")
        _, stdout, stderr = client.exec_command(req_install_cmd)
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        
        if stderr_result and "error" in stderr_result.lower():
            print(f"Error installing requirements: {stderr_result}")
            if stdout_result:
                print(f"Output: {stdout_result}")
            
            # Try installing critical packages directly even if requirements.txt installation fails
            print("Attempting to install critical packages directly...")
            packages = ["torch", "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "anndata", "scanpy", "scvi-tools"]
            direct_install_cmd = f"pip install {' '.join(packages)}"
            print(f"Running direct installation command: {direct_install_cmd}")
            _, stdout, stderr = client.exec_command(direct_install_cmd)
            stdout_result = stdout.read().decode()
            stderr_result = stderr.read().decode()
            
            if stderr_result and "error" in stderr_result.lower():
                print(f"Error installing critical packages: {stderr_result}")
                if stdout_result:
                    print(f"Output: {stdout_result}")
            else:
                print("Critical packages installed successfully.")
        else:
            print("All requirements installed successfully from requirements.txt")
        
        # Now install the project in development mode
        install_cmd = f"cd {remote_work_dir} && pip install -e ."
        print(f"Running project installation command: {install_cmd}")
        _, stdout, stderr = client.exec_command(install_cmd)
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        
        if stderr_result and "error" in stderr_result.lower():
            print(f"Error installing project: {stderr_result}")
            if stdout_result:
                print(f"Output: {stdout_result}")
        else:
            print("Project installed successfully in development mode.")
        
        # Build the command to run the benchmark
        cmd_parts = [
            f"cd {remote_work_dir}",
            f"python -m sccvi_impl.scripts.{args.script.replace('.py', '')}_benchmarking --n_epochs {args.epochs} --batch_size {args.batch_size} --skip_visualization"
        ]
        
        if has_gpu:
            cmd_parts[-1] += " --use_gpu"
        
        cmd = " && ".join(cmd_parts)
        
        # Run the benchmark
        print(f"Running remote benchmark: {cmd}")
        start_time = time.time()
        
        # Execute command and stream output
        _, stdout, stderr = client.exec_command(cmd)
        
        # Stream stdout in real-time
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                output = stdout.channel.recv(1024).decode()
                print(output, end="")
            time.sleep(0.1)
        
        # Get remaining output
        output = stdout.read().decode()
        if output:
            print(output, end="")
            
        # Get errors
        error = stderr.read().decode()
        if error:
            print(f"Errors:\n{error}")
        
        # Check exit status
        exit_status = stdout.channel.recv_exit_status()
        success = exit_status == 0
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Print summary
        print(f"\nRemote benchmark {'completed successfully' if success else 'failed'}")
        print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        return success
        
    except Exception as e:
        print(f"SSH connection error: {e}")
        return False
    finally:
        client.close()
        print("SSH connection closed.")

def main():
    args = parse_args()
    
    # Get the script path
    script_path = get_script_path(args.script)
    print(f"Selected benchmark: {args.script}")
    
    if args.remote:
        # Run the benchmark on remote host
        print("Running benchmark on remote JarvisLabs instance via SSH")
        success = run_benchmark_remote(args)
    else:
        # Run the benchmark locally
        print(f"Running benchmark locally")
        print(f"Script path: {script_path}")
        success = run_benchmark(
            script_path, 
            args.epochs, 
            args.batch_size,
            args.monitor,
            args.monitor_interval
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
