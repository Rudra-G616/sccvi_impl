import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_benchmark_results(results_dir: str) -> Dict:
    """
    Load benchmark results from the results directory.
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        Dictionary with all benchmark results
    """
    if not os.path.exists(results_dir):
        st.error(f"Results directory {results_dir} does not exist")
        return {}
    
    # Check if combined results file exists
    combined_path = os.path.join(results_dir, "all_benchmarks.json")
    if os.path.exists(combined_path):
        with open(combined_path, 'r') as f:
            return json.load(f)
    
    # If no combined file, try to load individual dataset results
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith("_benchmark.json"):
            dataset_name = filename.replace("_benchmark.json", "")
            with open(os.path.join(results_dir, filename), 'r') as f:
                results[dataset_name] = json.load(f)
    
    return results


def create_comparison_df(results: Dict) -> pd.DataFrame:
    """
    Create a DataFrame for easy comparison of metrics across models and datasets.
    
    Args:
        results: Dictionary with benchmark results
        
    Returns:
        DataFrame with metrics
    """
    data = []
    
    for dataset_name, dataset_results in results.items():
        for model_name in ["sccausalvi", "model1"]:
            if model_name in dataset_results:
                model_metrics = dataset_results[model_name]
                data.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "correlation_bg_te": model_metrics["correlation_bg_te"],
                    "silhouette_bg": model_metrics["silhouette_bg"],
                    "reconstruction_error": model_metrics["reconstruction_error"]
                })
    
    return pd.DataFrame(data)


def plot_metrics_comparison(df: pd.DataFrame):
    """
    Create bar plots comparing metrics between models across datasets.
    
    Args:
        df: DataFrame with metrics
    """
    metrics = [
        ("correlation_bg_te", "Correlation (E[z_bg*e_tilda] - E[z_bg]*E[e_tilda])"),
        ("silhouette_bg", "Average Silhouette Score on z_bg"),
        ("reconstruction_error", "L2 Reconstruction Error")
    ]
    
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[title for _, title in metrics],
        vertical_spacing=0.15
    )
    
    for i, (metric_name, _) in enumerate(metrics):
        model_colors = {"sccausalvi": "blue", "model1": "orange"}
        
        for model in ["sccausalvi", "model1"]:
            model_df = df[df["model"] == model]
            
            fig.add_trace(
                go.Bar(
                    x=model_df["dataset"],
                    y=model_df[metric_name],
                    name=model if i == 0 else None,  # Only add to legend once
                    marker_color=model_colors[model],
                    showlegend=(i == 0)  # Only show in legend for first subplot
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=200 * len(metrics),
        width=800,
        title_text="Model Comparison Across Datasets",
        legend_title_text="Model",
        barmode='group'
    )
    
    st.plotly_chart(fig)


def plot_radar_chart(df: pd.DataFrame):
    """
    Create radar charts comparing models for each dataset.
    
    Args:
        df: DataFrame with metrics
    """
    datasets = df["dataset"].unique()
    metrics = ["correlation_bg_te", "silhouette_bg", "reconstruction_error"]
    
    # For radar chart, we need to normalize metrics between 0 and 1
    # For reconstruction error, lower is better, so we invert it
    radar_df = df.copy()
    
    # Normalize each metric
    for metric in metrics:
        if metric == "reconstruction_error":
            # For reconstruction error, lower is better
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            if max_val > min_val:
                radar_df[metric] = 1 - ((radar_df[metric] - min_val) / (max_val - min_val))
        else:
            # For other metrics, higher is better
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            if max_val > min_val:
                radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
    
    # Create radar charts for each dataset
    for dataset in datasets:
        dataset_df = radar_df[radar_df["dataset"] == dataset]
        
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each model
        for model in ["sccausalvi", "model1"]:
            model_df = dataset_df[dataset_df["model"] == model]
            if not model_df.empty:
                fig.add_trace(go.Scatterpolar(
                    r=model_df[metrics].values[0],
                    theta=["Correlation", "Silhouette Score", "Reconstruction Accuracy"],
                    fill='toself',
                    name=model
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Model Comparison for {dataset} Dataset",
            showlegend=True
        )
        
        st.plotly_chart(fig)


def app():
    st.set_page_config(
        page_title="scCausalVI vs Model1 Benchmark",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🧬 scCausalVI vs Model1 Benchmark Dashboard")
    st.markdown("""
    This dashboard presents benchmark results comparing scCausalVI and Model1 on multiple datasets.
    The benchmarks evaluate three key metrics:
    
    1. **Correlation**: E[z_bg*e_tilda] - E[z_bg]*E[e_tilda]
    2. **Silhouette Score**: Average silhouette score on background latent space (z_bg)
    3. **Reconstruction Error**: L2 norm of difference between original and reconstructed data
    """)
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value=os.path.join("data", "benchmarking_results")
    )
    
    # Load benchmark results
    results = load_benchmark_results(results_dir)
    
    if not results:
        st.warning("No benchmark results found. Please run the benchmarking script first.")
        
        st.markdown("""
        ### How to run benchmarking
        
        You can run the benchmarking script with the following command:
        ```bash
        python -m sccvi_impl.benchmarking.run_benchmark
        ```
        
        Or to benchmark a specific dataset:
        ```bash
        python -m sccvi_impl.benchmarking.run_benchmark --dataset simulated
        ```
        """)
        return
    
    # Create DataFrame for visualization
    df = create_comparison_df(results)
    
    # Main layout
    tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Radar Charts", "Raw Data"])
    
    with tab1:
        st.header("Metric Comparison")
        plot_metrics_comparison(df)
        
        st.markdown("""
        #### Metrics Explanation
        - **Correlation**: Higher values indicate stronger correlation between background and treatment effect latent spaces
        - **Silhouette Score**: Higher values indicate better separation of conditions in the background latent space
        - **Reconstruction Error**: Lower values indicate better reconstruction of the original data
        """)
    
    with tab2:
        st.header("Radar Charts")
        st.markdown("""
        These radar charts show the relative performance of each model across all metrics for each dataset.
        All metrics are normalized to a 0-1 scale, with 1 being better.
        """)
        plot_radar_chart(df)
    
    with tab3:
        st.header("Raw Data")
        st.dataframe(df)
        
        # Download button for CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="benchmark_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    app()
