import os
import gdown
from typing import Dict, Optional

# Download datasets used in paper

# Dataset URLs mapping with clear, descriptive names
DATASET_URLS: Dict[str, str] = {
    "simulated": "1nc6vhO3ObZoubYUe4xZ3WCxwB6Way6oC",
    "ifn_beta": "1Im52_rfWWOzsINCETN-ObIJZRU5JFF_D",
    "covid_epithelial": "1bPwejbf6RGEwByzujHSpAkfhTrUOwNSN",
    "covid_pbmc": "1wuejNfaUKJw5ExaJh8s6WzqJO8b_cYjv",
    "pbmc_batch_effect": "19XqOr8odcPq4nBDhZWTV4y-uLc3y_L-e",
    "pbmc_negative_control": "1d6buhzJsc0qv2GTIwRHrXPApMQFaADq9"
}

def download_dataset(dataset_name: str, output_dir: str = "data") -> Optional[str]:
    """
    Download a single-cell dataset from Google Drive.
    
    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to save downloaded files
    
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct file paths
    file_id = DATASET_URLS[dataset_name]
    output_path = os.path.join(output_dir, f"{dataset_name}.h5ad")
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Dataset {dataset_name} already exists at {output_path}")
        return output_path
    
    # Construct Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Download file
        gdown.download(url, output_path, quiet=False)
        print(f"Successfully downloaded {dataset_name} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return None

def download_all_datasets(output_dir: str = "data") -> None:
    """Download all available datasets."""
    for dataset_name in DATASET_URLS:
        download_dataset(dataset_name, output_dir)

if __name__ == "__main__":
    download_all_datasets()