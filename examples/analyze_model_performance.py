#!/usr/bin/env python3
"""
Script to analyze model performance from CSV evaluation results.
Groups records by model file and calculates total success rate across all scenarios.
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from collections import defaultdict


def extract_model_name(model_path):
    """Extract model name from full path for better readability."""
    return os.path.basename(model_path)


def analyze_csv(csv_file, output_file=None):
    """
    Analyze CSV file and group by model file with success rate statistics.
    
    Args:
        csv_file (str): Path to the CSV file
        output_file (str, optional): Path to save output CSV
    
    Returns:
        pd.DataFrame: Grouped results with success rate statistics
    """
    
    # Read CSV file
    # Assuming columns based on the data structure observed
    column_names = [
        'algorithm', 'scenario', 'agents1', 'agents2', 
        'model_path', 'success_rate', 'metric2', 'metric3'
    ]
    
    try:
        df = pd.read_csv(csv_file, names=column_names)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    print(f"Loaded {len(df)} records from {csv_file}")
    print(f"Unique scenarios: {df['scenario'].unique()}")
    print(f"Unique models: {df['model_path'].nunique()}")
    
    # Add model name column for easier identification
    df['model_name'] = df['model_path'].apply(extract_model_name)
    
    # Group by model file and calculate statistics
    model_stats = []
    
    for model_path in df['model_path'].unique():
        model_data = df[df['model_path'] == model_path]
        model_name = extract_model_name(model_path)
        
        # Calculate statistics
        total_scenarios = len(model_data)
        mean_success_rate = model_data['success_rate'].mean()
        median_success_rate = model_data['success_rate'].median()
        min_success_rate = model_data['success_rate'].min()
        max_success_rate = model_data['success_rate'].max()
        std_success_rate = model_data['success_rate'].std()
        
        # Count scenarios where success rate > threshold (e.g., 0.5)
        high_success_scenarios = len(model_data[model_data['success_rate'] > 0.5])
        
        # Get scenario breakdown
        scenario_breakdown = model_data.groupby('scenario')['success_rate'].mean().to_dict()
        
        model_stats.append({
            'model_name': model_name,
            'model_path': model_path,
            'total_scenarios': total_scenarios,
            'mean_success_rate': mean_success_rate,
            'median_success_rate': median_success_rate,
            'min_success_rate': min_success_rate,
            'max_success_rate': max_success_rate,
            'std_success_rate': std_success_rate,
            'high_success_scenarios': high_success_scenarios,
            'high_success_ratio': high_success_scenarios / total_scenarios if total_scenarios > 0 else 0,
            **{f'success_rate_{scenario}': rate for scenario, rate in scenario_breakdown.items()}
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(model_stats)
    
    # Sort by mean success rate (descending)
    results_df = results_df.sort_values('mean_success_rate', ascending=False)
    
    # Display results
    print("\n" + "="*100)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*100)
    
    print(f"\nTop 10 Models by Mean Success Rate:")
    print("-" * 80)
    top_models = results_df.head(10)
    
    for idx, row in top_models.iterrows():
        print(f"\nModel: {row['model_name']}")
        print(f"  Mean Success Rate: {row['mean_success_rate']:.4f}")
        print(f"  Median Success Rate: {row['median_success_rate']:.4f}")
        print(f"  Range: {row['min_success_rate']:.4f} - {row['max_success_rate']:.4f}")
        print(f"  Std Dev: {row['std_success_rate']:.4f}")
        print(f"  High Success Scenarios: {row['high_success_scenarios']}/{row['total_scenarios']} ({row['high_success_ratio']:.2%})")
        
        # Show scenario breakdown
        scenario_cols = [col for col in row.index if col.startswith('success_rate_') and not pd.isna(row[col])]
        if scenario_cols:
            print(f"  Scenario Breakdown:")
            for col in scenario_cols:
                scenario_name = col.replace('success_rate_', '')
                print(f"    {scenario_name}: {row[col]:.4f}")
    
    # Summary statistics
    print(f"\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total models analyzed: {len(results_df)}")
    print(f"Best performing model: {results_df.iloc[0]['model_name']} (Mean: {results_df.iloc[0]['mean_success_rate']:.4f})")
    print(f"Worst performing model: {results_df.iloc[-1]['model_name']} (Mean: {results_df.iloc[-1]['mean_success_rate']:.4f})")
    print(f"Average success rate across all models: {results_df['mean_success_rate'].mean():.4f}")
    print(f"Models with >50% average success rate: {len(results_df[results_df['mean_success_rate'] > 0.5])}/{len(results_df)}")
    
    # Save results if output file specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results_df


def create_detailed_report(df, csv_file, output_dir=None):
    """Create a detailed report with scenario-wise analysis."""
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    # Scenario-wise analysis
    print(f"\n" + "="*100)
    print("SCENARIO-WISE ANALYSIS")
    print("="*100)
    
    scenario_stats = df.groupby('scenario').agg({
        'success_rate': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'model_path': 'nunique'
    }).round(4)
    
    scenario_stats.columns = ['num_evaluations', 'mean_success', 'median_success', 
                             'std_success', 'min_success', 'max_success', 'unique_models']
    
    print(scenario_stats)
    
    # Model training progress analysis (if model names contain iteration numbers)
    print(f"\n" + "="*100)
    print("MODEL TRAINING PROGRESS ANALYSIS")
    print("="*100)
    
    # Extract iteration numbers from model names
    df['model_iteration'] = df['model_name'].str.extract(r'(\d+)').astype(float)
    
    if not df['model_iteration'].isna().all():
        # Group by iteration and calculate mean success rate
        iteration_progress = df.groupby('model_iteration')['success_rate'].agg(['count', 'mean', 'std']).round(4)
        iteration_progress = iteration_progress.sort_index()
        
        print("Training Progress (by iteration):")
        print(iteration_progress.head(20))  # Show first 20 iterations
        
        # Save iteration progress
        if output_dir:
            iteration_file = os.path.join(output_dir, 'iteration_progress.csv')
            iteration_progress.to_csv(iteration_file)
            print(f"Iteration progress saved to: {iteration_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance from CSV evaluation results')
    parser.add_argument('csv_file', help='Path to the CSV file containing evaluation results')
    parser.add_argument('--output', '-o', help='Output CSV file for grouped results')
    parser.add_argument('--detailed', '-d', action='store_true', help='Create detailed analysis report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found")
        return 1
    
    # Read and analyze the CSV
    df = pd.read_csv(args.csv_file, names=[
        'algorithm', 'scenario', 'agents1', 'agents2', 
        'model_path', 'success_rate', 'metric2', 'metric3'
    ])
    
    # Perform analysis
    results_df = analyze_csv(args.csv_file, args.output)
    
    if results_df is None:
        return 1
    
    # Create detailed report if requested
    if args.detailed:
        create_detailed_report(df, args.csv_file)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
