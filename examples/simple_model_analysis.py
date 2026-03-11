#!/usr/bin/env python3
"""
Simple script to read CSV and group records by model file with total success rate.
"""

import pandas as pd
import os


def analyze_model_performance(csv_file):
    """
    Read CSV file and group by model file, calculating success rate statistics.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    
    # Define column names based on the CSV structure
    columns = ['algorithm', 'scenario', 'agents1', 'agents2', 'model_path', 'success_rate', 'metric2', 'metric3']
    
    # Read the CSV file
    df = pd.read_csv(csv_file, names=columns)
    
    print(f"Loaded {len(df)} records from {csv_file}")
    print(f"Unique scenarios: {sorted(df['scenario'].unique())}")
    print(f"Unique models: {df['model_path'].nunique()}")
    
    # Extract model name from path for easier reading
    df['model_name'] = df['model_path'].apply(lambda x: os.path.basename(x))
    
    # Group by model file and calculate statistics
    model_summary = df.groupby(['model_path', 'model_name']).agg({
        'success_rate': ['count', 'mean', 'sum', 'std', 'min', 'max'],
        'scenario': lambda x: list(x.unique())
    }).round(4)
    
    # Flatten column names
    model_summary.columns = ['num_scenarios', 'avg_success_rate', 'total_success_sum', 
                           'std_success_rate', 'min_success_rate', 'max_success_rate', 'scenarios_tested']
    
    # Calculate total success rate as percentage of successful scenarios
    model_summary['total_success_rate'] = model_summary['avg_success_rate']
    
    # Sort by average success rate (descending)
    model_summary = model_summary.sort_values('avg_success_rate', ascending=False)
    
    # Reset index to make model_path and model_name regular columns
    model_summary = model_summary.reset_index()
    
    # Display results
    print("\n" + "="*120)
    print("MODEL PERFORMANCE SUMMARY - GROUPED BY MODEL FILE")
    print("="*120)
    
    # Show summary table
    display_cols = ['model_name', 'num_scenarios', 'avg_success_rate', 'std_success_rate', 
                   'min_success_rate', 'max_success_rate']
    
    print(f"\n{len(model_summary)} unique models analyzed:")
    print("-" * 120)
    
    for idx, row in model_summary.iterrows():
        print(f"\nModel: {row['model_name']}")
        print(f"  Scenarios tested: {row['num_scenarios']}")
        print(f"  Average success rate: {row['avg_success_rate']:.4f} ({row['avg_success_rate']*100:.2f}%)")
        print(f"  Success rate range: {row['min_success_rate']:.4f} - {row['max_success_rate']:.4f}")
        print(f"  Standard deviation: {row['std_success_rate']:.4f}")
        print(f"  Scenarios: {', '.join(row['scenarios_tested'])}")
    
    # Summary statistics
    print(f"\n" + "="*120)
    print("OVERALL STATISTICS")
    print("="*120)
    print(f"Best model: {model_summary.iloc[0]['model_name']} (Avg: {model_summary.iloc[0]['avg_success_rate']:.4f})")
    print(f"Worst model: {model_summary.iloc[-1]['model_name']} (Avg: {model_summary.iloc[-1]['avg_success_rate']:.4f})")
    print(f"Overall average success rate: {model_summary['avg_success_rate'].mean():.4f}")
    print(f"Models with >50% success rate: {len(model_summary[model_summary['avg_success_rate'] > 0.5])}/{len(model_summary)}")
    
    # Save results to CSV
    output_file = csv_file.replace('.csv', '_model_summary.csv')
    model_summary.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return model_summary


def create_scenario_breakdown(csv_file):
    """
    Create a detailed breakdown showing success rate by model and scenario.
    """
    
    columns = ['algorithm', 'scenario', 'agents1', 'agents2', 'model_path', 'success_rate', 'metric2', 'metric3']
    df = pd.read_csv(csv_file, names=columns)
    df['model_name'] = df['model_path'].apply(lambda x: os.path.basename(x))
    
    # Create pivot table: models as rows, scenarios as columns
    pivot_table = df.pivot_table(
        index=['model_name', 'model_path'], 
        columns='scenario', 
        values='success_rate', 
        aggfunc='mean'
    ).round(4)
    
    # Add summary statistics
    pivot_table['avg_all_scenarios'] = pivot_table.mean(axis=1)
    pivot_table['num_scenarios'] = pivot_table.count(axis=1)
    
    # Sort by average across all scenarios
    pivot_table = pivot_table.sort_values('avg_all_scenarios', ascending=False)
    
    print(f"\n" + "="*120)
    print("DETAILED BREAKDOWN: SUCCESS RATE BY MODEL AND SCENARIO")
    print("="*120)
    print(pivot_table)
    
    # Save detailed breakdown
    output_file = csv_file.replace('.csv', '_detailed_breakdown.csv')
    pivot_table.to_csv(output_file)
    print(f"\nDetailed breakdown saved to: {output_file}")
    
    return pivot_table


if __name__ == "__main__":
    csv_file = "/project/HARL-main/examples/HPN2H_Cube_10v10.csv"
    
    print("Analyzing model performance...")
    model_summary = analyze_model_performance(csv_file)
    
    print("\nCreating detailed scenario breakdown...")
    detailed_breakdown = create_scenario_breakdown(csv_file)
