#!/usr/bin/env python3
"""
Simple script to group CSV records by model file and calculate total success rate.
Usage: python group_by_model.py <csv_file>
"""

import pandas as pd
import sys
import os


def group_by_model_file(csv_file):
    """
    Read CSV and group by model file with success rate summary.
    """
    
    # Define column names
    columns = ['algorithm', 'scenario', 'agents1', 'agents2', 'model_path', 'success_rate', 'metric2', 'metric3']
    
    # Read CSV
    df = pd.read_csv(csv_file, names=columns)
    
    # Extract model filename
    df['model_file'] = df['model_path'].apply(os.path.basename)
    
    # Group by model file and calculate summary statistics
    summary = df.groupby('model_file').agg({
        'success_rate': ['count', 'mean', 'std', 'min', 'max'],
        'scenario': lambda x: len(x.unique())
    }).round(4)
    
    # Flatten column names
    summary.columns = ['num_evaluations', 'avg_success_rate', 'std_success_rate', 
                      'min_success_rate', 'max_success_rate', 'num_scenarios']
    
    # Sort by average success rate
    summary = summary.sort_values('avg_success_rate', ascending=False)
    
    # Add percentage format
    summary['success_percentage'] = (summary['avg_success_rate'] * 100).round(2)
    
    # Display results
    print(f"Model Performance Summary - {len(summary)} models analyzed")
    print("=" * 80)
    print(f"{'Model File':<25} {'Scenarios':<10} {'Avg Success':<12} {'Success %':<10} {'Range':<15}")
    print("-" * 80)
    
    for model, row in summary.iterrows():
        range_str = f"{row['min_success_rate']:.3f}-{row['max_success_rate']:.3f}"
        print(f"{model:<25} {row['num_scenarios']:<10} {row['avg_success_rate']:<12.4f} {row['success_percentage']:<10.1f}% {range_str:<15}")
    
    # Save summary
    output_file = csv_file.replace('.csv', '_grouped_by_model.csv')
    summary.to_csv(output_file)
    print(f"\nSummary saved to: {output_file}")
    
    # Quick stats
    print(f"\nQuick Statistics:")
    print(f"Best model: {summary.index[0]} ({summary.iloc[0]['success_percentage']:.1f}%)")
    print(f"Worst model: {summary.index[-1]} ({summary.iloc[-1]['success_percentage']:.1f}%)")
    print(f"Average across all models: {summary['success_percentage'].mean():.1f}%")
    print(f"Models with >50% success: {len(summary[summary['avg_success_rate'] > 0.5])}/{len(summary)}")
    
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python group_by_model.py <csv_file>")
        csv_file = "/project/HARL-main/examples/HPN2H_Cube_4v4.csv"  # Default for testing
    else:
        csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    summary = group_by_model_file(csv_file)
