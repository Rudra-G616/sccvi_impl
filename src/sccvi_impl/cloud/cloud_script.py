import os
import subprocess
import argparse
from pathlib import Path

def run_command(command):
    """Run a shell command and return its output."""
    print(f"\nRunning command: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if stdout.strip():
            print(f"Command output:\n{stdout}")
        if stderr.strip():
            print(f"Command error output:\n{stderr}")
        return stdout, stderr, process.returncode
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        raise

def parse_ssh_string(ssh_string):
    """Parse SSH string to extract host, port, and user information."""
    # Example: ssh -o StrictHostKeyChecking=no -p 12014 root@ssha.jarvislabs.ai
    parts = ssh_string.split()
    port = parts[parts.index('-p') + 1] if '-p' in parts else '22'
    user_host = parts[-1]  # last part contains user@host
    user, host = user_host.split('@')
    return user, host, port

def run_remote_command(command, ssh_string):
    """Run a command on the remote server."""
    user, host, port = parse_ssh_string(ssh_string)
    ssh_command = f'ssh -o StrictHostKeyChecking=no -p {port} {user}@{host} "{command}"'
    return run_command(ssh_command)

def setup_cloud_environment(ssh_string):
    """Set up the cloud environment with required packages."""
    commands = [
        "cd /root/sccvi_impl && "  # Ensure we're in the right directory
        "pip install -e . && "      # Install package in development mode
        "pip install -r requirements.txt"  # Install requirements
    ]
    
    stdout, stderr, code = run_remote_command(commands[0], ssh_string)
    if code != 0:
        raise Exception(f"Environment setup failed\nError: {stderr}")
    print("Environment setup completed")
    print(stdout)

def run_benchmarking(ssh_string, n_epochs=100, use_gpu=True, batch_size=128):
    """Run the simulated benchmarking script."""
    # Use python -m to run as a module, which fixes relative imports
    cmd = (
        "cd /root/sccvi_impl && "
        "PYTHONPATH=/root/sccvi_impl "
        f"python -m sccvi_impl.scripts.simulated_benchmarking "
        f"--n_epochs {n_epochs} "
        f"--batch_size {batch_size} "
        "--skip_visualization "  # Skip visualization on remote
        f"{'--use_gpu' if use_gpu else ''}"
    )
    stdout, stderr, code = run_remote_command(cmd, ssh_string)
    if code != 0:
        raise Exception(f"Benchmarking failed\nError: {stderr}")
    print("Benchmarking completed successfully")
    print(stdout)

def download_results(ssh_string):
    """Download the results file from the cloud."""
    cloud_results = "/root/sccvi_impl/sccvi_impl/src/sccvi_impl/results/simulated/benchmarking_results.csv"
    local_results = str(Path(__file__).parent.parent / "results/simulated/benchmarking_results.csv")
    
    # First verify the remote file exists
    verify_cmd = f"test -f {cloud_results} && echo 'File exists'"
    stdout, stderr, code = run_remote_command(verify_cmd, ssh_string)
    if code != 0:
        raise Exception(f"Results file not found on remote server: {cloud_results}")
    
    # Create the local directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(local_results), exist_ok=True)
    except Exception as e:
        raise Exception(f"Failed to create local directory: {str(e)}")
    
    user, host, port = parse_ssh_string(ssh_string)
    # Download results file from cloud
    scp_command = f"scp -P {port} {user}@{host}:{cloud_results} {local_results}"
    stdout, stderr, code = run_command(scp_command)
    if code != 0:
        raise Exception(f"Failed to download results\nError: {stderr}")
    
    # Verify local file exists after download
    if not os.path.exists(local_results):
        raise Exception("Download appeared to succeed but file not found locally")
        
    print(f"Results downloaded to: {local_results}")

def upload_files(ssh_string):
    """Upload project files to the cloud instance."""
    local_path = Path(__file__).parent.parent.parent.parent  # Get the root project directory
    user, host, port = parse_ssh_string(ssh_string)
    cloud_destination = f"{user}@{host}:/root"
    
    # Make sure destination directory exists and is clean
    mkdir_cmd = "rm -rf /root/sccvi_impl && mkdir -p /root/sccvi_impl"
    stdout, stderr, code = run_remote_command(mkdir_cmd, ssh_string)
    if code != 0:
        raise Exception(f"Failed to create remote directory\nError: {stderr}")
    
    # Upload the entire project directory itself, not just its contents
    scp_command = f"scp -P {port} -r {local_path} {cloud_destination}"
    stdout, stderr, code = run_command(scp_command)
    if code != 0:
        raise Exception(f"File upload failed\nError: {stderr}")
    
    # Verify pyproject.toml exists on remote
    verify_cmd = "ls -la /root/sccvi_impl/pyproject.toml"
    stdout, stderr, code = run_remote_command(verify_cmd, ssh_string)
    if code != 0:
        raise Exception(f"pyproject.toml not found after upload\nError: {stderr}")
    
    print("Files uploaded successfully")

def cleanup_remote(ssh_string):
    """Clean up any existing files on the remote server."""
    cmd = "rm -rf /root/sccvi_impl"
    stdout, stderr, code = run_remote_command(cmd, ssh_string)
    if code != 0:
        print(f"Warning: Could not clean up remote directory\nError: {stderr}")
    else:
        print("Cleaned up remote directory")

def main():
    parser = argparse.ArgumentParser(description='Run benchmarking on cloud GPU')
    parser.add_argument('--ssh-string', type=str, required=True,
                      help='SSH connection string (e.g., "ssh -o StrictHostKeyChecking=no -p 12014 root@ssha.jarvislabs.ai")')
    parser.add_argument('--n_epochs', type=int, default=100,
                      help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    parser.add_argument('--no-gpu', action='store_true',
                      help='Disable GPU usage even if available')
    args = parser.parse_args()
    
    try:
        print("Cleaning up remote directory...")
        cleanup_remote(args.ssh_string)
        
        print("Uploading files to cloud instance...")
        upload_files(args.ssh_string)
        
        print("Setting up cloud environment...")
        setup_cloud_environment(args.ssh_string)
        
        print("Running benchmarking...")
        run_benchmarking(args.ssh_string, 
                        n_epochs=args.n_epochs,
                        use_gpu=not args.no_gpu,
                        batch_size=args.batch_size)
        
        print("Downloading results...")
        download_results(args.ssh_string)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()


