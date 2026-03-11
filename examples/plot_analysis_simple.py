import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv('/project/HARL-main/examples/render1.csv', header=None)

# Define column names based on the provided information
columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
           'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg']
df.columns = columns

# Convert numeric columns
df['num_agents'] = pd.to_numeric(df['num_agents'])
df['goal_num'] = pd.to_numeric(df['goal_num'])
df['SR_avg'] = pd.to_numeric(df['SR_avg'])
df['success_episode_step_avg'] = pd.to_numeric(df['success_episode_step_avg'])

# Replace 9999 with NaN for success_episode_step_avg (indicating no success)
df['success_episode_step_avg'] = df['success_episode_step_avg'].replace(9999, np.nan)

# Filter: when goal_num = 100, only include num_agents <= 60
print(f"Original data shape: {df.shape}")
df = df[~((df['goal_num'] == 100) & (df['num_agents'] > 60))]
print(f"Filtered data shape: {df.shape}")
print("Filter applied: goal_num=100 limited to num_agents<=60")

Assignments=df['assignment'].unique()
Metrics=df['metric'].unique()
print("Unique agent numbers:", sorted(df['num_agents'].unique()))
print("Unique goal numbers:", sorted(df['goal_num'].unique()))
print("Assignments:", Assignments)
print("Metrics:", Metrics)

# Create the four plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Analysis: Success Rate and Episode Steps', fontsize=16, fontweight='bold')

# Colors for different method combinations
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
markers = ['o', 's', '^', 'D', 'v', '<']

# Plot 1: Fixed agent num = 5, SR vary with goal num
print("\nPlot 1: Fixed agents=5, SR vs goal_num")
data_1 = df[df['num_agents'] == 5]
grouped_1 = data_1.groupby(['goal_num', 'assignment', 'metric'])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in Assignments:
    for metric in Metrics:
        subset = grouped_1[(grouped_1['assignment'] == assignment) & 
                          (grouped_1['metric'] == metric)]
        if len(subset) > 0:
            axes[0,0].plot(subset['goal_num'].values, subset['SR_avg'].values, 
                          marker=markers[color_idx % len(markers)], 
                          color=colors[color_idx % len(colors)],
                          label=f'{assignment}-{metric}', linewidth=2, markersize=6)
            color_idx += 1

axes[0,0].set_xlabel('Goal Number', fontweight='bold')
axes[0,0].set_ylabel('Success Rate (SR)', fontweight='bold')
axes[0,0].set_title('Fixed Agents=5: Success Rate vs Goal Number', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_xlim(left=0)
axes[0,0].set_ylim(0, 1.05)

# Plot 2: Fixed agent num = 5, success_episode_step_avg vary with goal num
print("Plot 2: Fixed agents=5, success_episode_step_avg vs goal_num")
data_2 = df[(df['num_agents'] == 5) & (~df['success_episode_step_avg'].isna())]
grouped_2 = data_2.groupby(['goal_num', 'assignment', 'metric'])['success_episode_step_avg'].mean().reset_index()

color_idx = 0
for assignment in Assignments:
    for metric in Metrics:
        subset = grouped_2[(grouped_2['assignment'] == assignment) & 
                          (grouped_2['metric'] == metric)]
        if len(subset) > 0:
            axes[0,1].plot(subset['goal_num'].values, subset['success_episode_step_avg'].values, 
                          marker=markers[color_idx % len(markers)], 
                          color=colors[color_idx % len(colors)],
                          label=f'{assignment}-{metric}', linewidth=2, markersize=6)
            color_idx += 1

axes[0,1].set_xlabel('Goal Number', fontweight='bold')
axes[0,1].set_ylabel('Success Episode Steps (avg)', fontweight='bold')
axes[0,1].set_title('Fixed Agents=5: Success Episode Steps vs Goal Number', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xlim(left=0)

# Plot 3: Fixed goal num = 100, SR vary with agent num
print("Plot 3: Fixed goals=100, SR vs agent_num")
data_3 = df[df['goal_num'] == 100]
grouped_3 = data_3.groupby(['num_agents', 'assignment', 'metric'])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in Assignments:
    for metric in Metrics:
        subset = grouped_3[(grouped_3['assignment'] == assignment) & 
                          (grouped_3['metric'] == metric)]
        if len(subset) > 0:
            axes[1,0].plot(subset['num_agents'].values, subset['SR_avg'].values, 
                          marker=markers[color_idx % len(markers)], 
                          color=colors[color_idx % len(colors)],
                          label=f'{assignment}-{metric}', linewidth=2, markersize=6)
            color_idx += 1

axes[1,0].set_xlabel('Number of Agents', fontweight='bold')
axes[1,0].set_ylabel('Success Rate (SR)', fontweight='bold')
axes[1,0].set_title('Fixed Goals=100: Success Rate vs Number of Agents', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_xlim(left=0)
axes[1,0].set_ylim(0, 1.05)

# Plot 4: Fixed goal num = 100, success_episode_step_avg vary with agent num
print("Plot 4: Fixed goals=100, success_episode_step_avg vs agent_num")
data_4 = df[(df['goal_num'] == 100) & (~df['success_episode_step_avg'].isna())]
grouped_4 = data_4.groupby(['num_agents', 'assignment', 'metric'])['success_episode_step_avg'].mean().reset_index()

color_idx = 0
for assignment in Assignments:
    for metric in Metrics:
        subset = grouped_4[(grouped_4['assignment'] == assignment) & 
                          (grouped_4['metric'] == metric)]
        if len(subset) > 0:
            axes[1,1].plot(subset['num_agents'].values, subset['success_episode_step_avg'].values, 
                          marker=markers[color_idx % len(markers)], 
                          color=colors[color_idx % len(colors)],
                          label=f'{assignment}-{metric}', linewidth=2, markersize=6)
            color_idx += 1

axes[1,1].set_xlabel('Number of Agents', fontweight='bold')
axes[1,1].set_ylabel('Success Episode Steps (avg)', fontweight='bold')
axes[1,1].set_title('Fixed Goals=100: Success Episode Steps vs Number of Agents', fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_xlim(left=0)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('/project/HARL-main/examples/performance_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to performance_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\n1. Fixed Agents=5 (SR vs Goal Number):")
summary_1 = data_1.groupby('goal_num')['SR_avg'].agg(['mean', 'std', 'count'])
print(summary_1.head(10))

print(f"\n2. Fixed Agents=5 (Episode Steps vs Goal Number):")
summary_2 = data_2.groupby('goal_num')['success_episode_step_avg'].agg(['mean', 'std', 'count'])
print(summary_2.head(10))

print(f"\n3. Fixed Goals=100 (SR vs Agent Number):")
summary_3 = data_3.groupby('num_agents')['SR_avg'].agg(['mean', 'std', 'count'])
print(summary_3)

print(f"\n4. Fixed Goals=100 (Episode Steps vs Agent Number):")
summary_4 = data_4.groupby('num_agents')['success_episode_step_avg'].agg(['mean', 'std', 'count'])
print(summary_4)

print("\n" + "="*60)
print("ANALYSIS COMPLETE - Plots saved to performance_analysis.png")
print("="*60)
