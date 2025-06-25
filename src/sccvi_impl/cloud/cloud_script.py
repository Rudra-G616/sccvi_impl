"""Contains the code to run simulated_benchmarking.py in a cloud environment."""

import os
import subprocess
import argparse
from pathlib import Path
import sys


def run_ssh_command(command, ssh_string):
    """Run a command on the remote server via SSH.
    
    Args:
        command: The command to run on the remote server.
        ssh_string: The SSH connection string.
        
    Returns:
        The command output.
    """
    full_command = f"{ssh_string} '{command}'"
    print(f"Executing: {full_command}")
    
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}", file=sys.stderr)
        return None
    
    return result.stdout.strip()


def determine_python_command(ssh_string):
    """Determine the Python command to use (python or python3).
    
    Args:
        ssh_string: The SSH connection string.
        
    Returns:
        The Python command to use.
    """
    # Try python first
    python_check = run_ssh_command("which python", ssh_string)
    if python_check:
        # Verify it's Python 3.x
        version_check = run_ssh_command("python --version", ssh_string)
        if version_check and "Python 3" in version_check:
            return "python"
    
    # Try python3
    python3_check = run_ssh_command("which python3", ssh_string)
    if python3_check:
        return "python3"
    
    # No Python 3 found, try to install it
    print("Python 3 not found, installing...")
    run_ssh_command("apt-get update && apt-get install -y python3 python3-pip", ssh_string)
    
    # Check again
    python3_check = run_ssh_command("which python3", ssh_string)
    if python3_check:
        return "python3"
    
    print("Failed to install Python 3. Please install it manually.")
    return None


def determine_pip_command(ssh_string, python_cmd):
    """Determine the pip command to use (pip or pip3).
    
    Args:
        ssh_string: The SSH connection string.
        python_cmd: The Python command to use.
        
    Returns:
        The pip command to use.
    """
    # Try pip
    pip_check = run_ssh_command("which pip", ssh_string)
    if pip_check:
        # Verify it's for Python 3.x
        version_check = run_ssh_command("pip --version", ssh_string)
        if version_check and "python 3" in version_check.lower():
            return "pip"
    
    # Try pip3
    pip3_check = run_ssh_command("which pip3", ssh_string)
    if pip3_check:
        return "pip3"
    
    # Try to install pip using the Python command
    print(f"pip for Python 3 not found, installing using {python_cmd}...")
    run_ssh_command(f"{python_cmd} -m ensurepip --upgrade", ssh_string)
    
    # Check again for pip
    pip_check = run_ssh_command(f"which pip", ssh_string)
    if pip_check:
        version_check = run_ssh_command("pip --version", ssh_string)
        if version_check and "python 3" in version_check.lower():
            return "pip"
    
    # Check again for pip3
    pip3_check = run_ssh_command("which pip3", ssh_string)
    if pip3_check:
        return "pip3"
    
    # Use the Python module as a last resort
    return f"{python_cmd} -m pip"


def upload_file(local_path, remote_path, ssh_string):
    """Upload a file to the remote server using SCP.
    
    Args:
        local_path: Path to the local file.
        remote_path: Path where to place the file on the remote server.
        ssh_string: The SSH connection string.
        
    Returns:
        True if the upload was successful, False otherwise.
    """
    # Extract the SSH parameters from the SSH string
    parts = ssh_string.split()
    port = next((parts[i+1] for i, part in enumerate(parts) if part == "-p"), "22")
    host = parts[-1]
    
    scp_command = f"scp -o StrictHostKeyChecking=no -P {port} {local_path} {host}:{remote_path}"
    print(f"Uploading: {scp_command}")
    
    result = subprocess.run(scp_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error uploading file: {result.stderr}", file=sys.stderr)
        return False
    
    return True


def setup_remote_environment(ssh_string, local_root_dir):
    """Set up the remote environment with required dependencies.
    
    Args:
        ssh_string: The SSH connection string.
        local_root_dir: Path to the local project root directory.
        
    Returns:
        True if setup was successful, False otherwise.
    """
    # Determine the Python command to use (python or python3)
    python_cmd = determine_python_command(ssh_string)
    if not python_cmd:
        return False
    
    # Determine the pip command to use (pip or pip3)
    pip_cmd = determine_pip_command(ssh_string, python_cmd)
    if not pip_cmd:
        return False
    
    # Create project directory
    print("Creating project directory...")
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl", ssh_string)
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl/scripts", ssh_string)
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl/model", ssh_string)
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl/module", ssh_string)
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl/data", ssh_string)
    
    # Upload requirements.txt
    local_requirements = os.path.join(local_root_dir, "requirements.txt")
    if not upload_file(local_requirements, "~/sccvi_project/requirements.txt", ssh_string):
        print("Failed to upload requirements.txt")
        return False
    
    # Upload pyproject.toml
    local_pyproject = os.path.join(local_root_dir, "pyproject.toml")
    if not upload_file(local_pyproject, "~/sccvi_project/pyproject.toml", ssh_string):
        print("Failed to upload pyproject.toml")
        return False
    
    # Upload package __init__.py files
    for path in ["src/sccvi_impl/__init__.py", "src/sccvi_impl/scripts/__init__.py", 
                 "src/sccvi_impl/model/__init__.py", "src/sccvi_impl/module/__init__.py",
                 "src/sccvi_impl/data/__init__.py"]:
        local_path = os.path.join(local_root_dir, path)
        remote_path = f"~/sccvi_project/{path}"
        if os.path.exists(local_path) and not upload_file(local_path, remote_path, ssh_string):
            print(f"Failed to upload {path}")
            return False
    
    # Install requirements from requirements.txt
    print("Installing requirements from requirements.txt...")
    run_ssh_command(f"cd ~/sccvi_project && {pip_cmd} install -r requirements.txt", ssh_string)
    
    # Install package in development mode
    print("Installing package in development mode...")
    run_ssh_command(f"cd ~/sccvi_project && {pip_cmd} install -e .", ssh_string)
    
    return True


def run_benchmarking(local_benchmark_path, ssh_string, local_root_dir):
    """Run the benchmarking script on the remote server.
    
    Args:
        local_benchmark_path: Path to the local benchmarking script.
        ssh_string: The SSH connection string.
        local_root_dir: Path to the local project root directory.
        
    Returns:
        True if benchmarking was successful, False otherwise.
    """
    # Setup remote environment
    if not setup_remote_environment(ssh_string, local_root_dir):
        return False
    
    # Get the Python command to use
    python_cmd = determine_python_command(ssh_string)
    if not python_cmd:
        return False
    
    # Upload required model files
    for module in ["model", "module", "data"]:
        src_dir = os.path.join(local_root_dir, f"src/sccvi_impl/{module}")
        if os.path.exists(src_dir):
            for filename in os.listdir(src_dir):
                if filename.endswith(".py"):
                    local_file = os.path.join(src_dir, filename)
                    remote_file = f"~/sccvi_project/src/sccvi_impl/{module}/{filename}"
                    if not upload_file(local_file, remote_file, ssh_string):
                        print(f"Failed to upload {local_file}")
                        return False
    
    # Upload benchmarking script
    remote_path = "~/sccvi_project/src/sccvi_impl/scripts/simulated_benchmarking.py"
    if not upload_file(local_benchmark_path, remote_path, ssh_string):
        return False
    
    # Run the benchmarking script
    print("Running benchmarking script...")
    output = run_ssh_command(f"cd ~/sccvi_project && {python_cmd} -m src.sccvi_impl.scripts.simulated_benchmarking", ssh_string)
    
    if output:
        print("Benchmarking results:")
        print(output)
    
    # Download results if they were saved to files
    download_results(ssh_string, local_root_dir)
    
    return True


def download_results(ssh_string, local_root_dir):
    """Download benchmark results from the remote server.
    
    Args:
        ssh_string: The SSH connection string.
        local_root_dir: Path to the local project root directory.
    """
    # Extract the SSH parameters from the SSH string
    parts = ssh_string.split()
    port = next((parts[i+1] for i, part in enumerate(parts) if part == "-p"), "22")
    host = parts[-1]
    
    # Create local results directory if it doesn't exist
    local_results_dir = os.path.join(local_root_dir, "src/sccvi_impl/results/simulated")
    os.makedirs(local_results_dir, exist_ok=True)
    
    # Download benchmark_results.csv if it exists
    remote_results = "~/sccvi_project/src/sccvi_impl/results/simulated/benchmark_results.csv"
    local_results = os.path.join(local_results_dir, "benchmark_results.csv")
    
    # Make sure remote results directory exists
    run_ssh_command("mkdir -p ~/sccvi_project/src/sccvi_impl/results/simulated", ssh_string)
    
    print(f"Checking for results file: {remote_results}")
    check_results = run_ssh_command(f"test -f {remote_results} && echo 'exists' || echo 'not found'", ssh_string)
    
    if check_results and "exists" in check_results:
        print("Downloading benchmark results...")
        scp_command = f"scp -o StrictHostKeyChecking=no -P {port} {host}:{remote_results} {local_results}"
        result = subprocess.run(scp_command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Results downloaded to {local_results}")
        else:
            print(f"Error downloading results: {result.stderr}", file=sys.stderr)
    else:
        print("No benchmark results file found on the remote server")


def main():
    """Main function to parse arguments and run the benchmarking."""
    parser = argparse.ArgumentParser(description="Run simulated benchmarking in a cloud environment.")
    parser.add_argument("--benchmark-script", type=str, default=None, 
                        help="Path to the simulated_benchmarking.py script.")
    parser.add_argument("--ssh-string", type=str, required=True,
                        help="SSH connection string for the remote server.")
    
    args = parser.parse_args()
    
    # Find the project root directory
    current_dir = Path(__file__).resolve().parent
    project_root = None
    
    # Try to find the project root by looking for pyproject.toml
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    if project_root is None:
        print("Error: Could not find project root (directory containing pyproject.toml)")
        return 1
    
    # If no benchmark script is provided, try to find it
    if args.benchmark_script is None:
        script_path = project_root / "src" / "sccvi_impl" / "scripts" / "simulated_benchmarking.py"
        
        if script_path.exists():
            args.benchmark_script = str(script_path)
        else:
            print(f"Error: Could not find simulated_benchmarking.py at {script_path}.")
            print("Please provide the path using --benchmark-script.")
            return 1
    
    print(f"Using benchmark script: {args.benchmark_script}")
    print(f"Project root: {project_root}")
    
    success = run_benchmarking(args.benchmark_script, args.ssh_string, str(project_root))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

