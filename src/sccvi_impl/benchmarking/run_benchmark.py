import os
import sys
import argparse

from .model_benchmark import ModelBenchmarker

def main(args):
    """Run benchmarking on specified datasets"""
    print("Starting model benchmarking...")
    
    benchmarker = ModelBenchmarker(
        output_dir=args.output_dir,
        n_latent=args.n_latent,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        use_gpu=args.use_gpu
    )
    
    if args.dataset:
        # Benchmark a specific dataset
        print(f"Benchmarking on dataset: {args.dataset}")
        benchmarker.train_models(args.dataset)
    else:
        # Benchmark all datasets
        print("Benchmarking on all datasets")
        benchmarker.benchmark_all_datasets()
    
    print(f"Benchmarking complete. Results saved to {os.path.join(args.output_dir, 'benchmarking_results')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scCausalVI and Model1 on datasets")
    parser.add_argument("--dataset", type=str, help="Specific dataset to benchmark (default: all)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save datasets and results")
    parser.add_argument("--n_latent", type=int, default=15, help="Dimension of latent space (used for both models)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    
    args = parser.parse_args()
    main(args)
