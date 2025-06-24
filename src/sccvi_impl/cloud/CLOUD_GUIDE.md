# scCausalVI Cloud Execution Guide

This guide provides instructions for running scCausalVI benchmarking in cloud environments, specifically on JarvisLabs GPU instances.

## Overview

The `jarvis_gpu_runner.py` script supports both local and remote execution of scCausalVI benchmarking scripts. Remote execution leverages SSH to connect to JarvisLabs GPU instances, upload necessary files, and run benchmarks.

## Requirements

- Python 3.8+
- paramiko (for SSH connectivity)
- A JarvisLabs account with GPU instance(s)
- SSH access to your JarvisLabs instance

## Installation

1. Make sure you have the required dependencies

2. Set up SSH access to your JarvisLabs instance:
   - Generate an SSH key pair if you don't have one already: `ssh-keygen -t rsa -b 4096`
   - Add your public key to JarvisLabs: Copy the contents of `~/.ssh/id_rsa.pub` to the JarvisLabs SSH key settings in your account dashboard
   - Test your connection: `ssh <username>@<your-instance>.jarviscloud.com`

## Usage

### Local Execution

To run benchmarks on your local machine:

```bash
python jarvis_gpu_runner.py --script simulated --epochs 200 --batch_size 256
```

### Remote Execution (SSH)

To run benchmarks on a remote JarvisLabs instance:

```bash
python jarvis_gpu_runner.py --remote \
    --host <your-instance>.jarviscloud.com \
    --username <username> \
    --script simulated \
    --epochs 200 \
    --batch_size 256
```

If your SSH key requires a passphrase, you'll be prompted to enter it.

To use a specific SSH key:

```bash
python jarvis_gpu_runner.py --remote \
    --host <your-instance>.jarviscloud.com \
    --username <username> \
    --key /path/to/your/private_key \
    --script simulated
```

For password-based authentication (not recommended):

```bash
python jarvis_gpu_runner.py --remote \
    --host <your-instance>.jarviscloud.com \
    --username <username> \
    --password \
    --script simulated
```

You'll be prompted to enter your password.

## Available Scripts

The following benchmarking scripts are available:

- `simulated`: Runs benchmarking on simulated data
- `covid_epithelial`: Runs benchmarking on COVID-19 epithelial cell data
- `covid_pbmc`: Runs benchmarking on COVID-19 PBMC data
- `ifn_beta`: Runs benchmarking on interferon beta-stimulated data

## Command-Line Options

### General Options

- `--script`: Which benchmarking script to run (required)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 128)
- `--monitor`: Enable GPU monitoring during training
- `--monitor_interval`: Interval in seconds between GPU monitoring checks (default: 30)

### SSH Options

- `--remote`: Run benchmark on a remote JarvisLabs instance via SSH
- `--host`: Hostname or IP address of the remote JarvisLabs instance
- `--port`: SSH port for the remote JarvisLabs instance (default: 22)
- `--username`: SSH username for the remote JarvisLabs instance
- `--key`: Path to SSH private key file (default: ~/.ssh/id_rsa)
- `--password`: Use password authentication instead of key-based authentication

## Common Issues and Troubleshooting

### SSH Connection Issues

1. **Permission denied errors**:
   - Make sure your SSH key is correctly added to JarvisLabs
   - Check file permissions: `chmod 600 ~/.ssh/id_rsa`

2. **Host key verification failed**:
   - If the host key has changed, remove the old entry: `ssh-keygen -R <hostname>`

3. **Connection timeout**:
   - Check if your JarvisLabs instance is running
   - Verify network connectivity and firewall settings

### GPU Issues

1. **GPU not detected**:
   - Make sure your JarvisLabs instance has a GPU attached
   - Check nvidia-smi output for GPU status: `nvidia-smi`

2. **CUDA out of memory**:
   - Reduce batch size using the `--batch_size` parameter
   - Consider using a JarvisLabs instance with a larger GPU

## Best Practices

1. **Use key-based authentication** instead of passwords for better security
2. **Monitor GPU usage** with the `--monitor` flag to track resource utilization
3. **Adjust batch size** based on available GPU memory
4. **Use screen or tmux** on the remote host for long-running benchmarks to maintain sessions even if your connection drops

## JarvisLabs-Specific Tips

1. **Instance Types**: Choose the appropriate GPU instance type based on your computational needs
2. **Storage**: Make sure your JarvisLabs instance has enough storage for your datasets
3. **Cost Management**: Remember to stop your instances when not in use to avoid unnecessary charges
4. **Persistent Storage**: Use JarvisLabs persistent storage options for important data

## Further Resources

- [JarvisLabs Documentation](https://jarvislabs.ai/docs/)
- [scCausalVI Documentation](https://github.com/yourusername/sccvi_impl)
- [Paramiko SSH Documentation](http://docs.paramiko.org/)
