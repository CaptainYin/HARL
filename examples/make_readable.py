#!/usr/bin/env python3
import pandas as pd

def create_readable_version(input_file, output_file):
    """Create a more readable version with better column names"""
    
    # Read the reorganized CSV
    df = pd.read_csv(input_file)
    
    # Rename columns for better readability
    column_mapping = {
        'model_name': 'Model Name',
        'success_rate_CA': 'CA Success Rate',
        'success_rate_Cube': 'Cube Success Rate', 
        'success_rate_Unmaze': 'Unmaze Success Rate',
        'success_rate_Warehouse': 'Warehouse Success Rate',
        'success_rate_scenario1': 'Scenario1 Success Rate',
        'success_rate_scenario3': 'Scenario3 Success Rate',
        'success_rate_scenario6': 'Scenario6 Success Rate'
    }
    
    df_readable = df.rename(columns=column_mapping)
    
    # Round success rates to 4 decimal places for better readability
    for col in df_readable.columns:
        if 'Success Rate' in col:
            df_readable[col] = df_readable[col].round(4)
    
    # Save the readable version
    df_readable.to_csv(output_file, index=False)
    
    print(f"Readable version saved to {output_file}")
    print(f"\nFirst 10 rows of readable version:")
    print(df_readable.head(10).to_string(index=False))
    
    return df_readable

if __name__ == "__main__":
    input_file = "rendermix_reorganized.csv"
    output_file = "rendermix_final.csv"
    
    df_readable = create_readable_version(input_file, output_file)
    print(f"\nTotal models: {len(df_readable)}")
    print(f"Scenarios covered: {[col.replace(' Success Rate', '') for col in df_readable.columns if 'Success Rate' in col]}")
