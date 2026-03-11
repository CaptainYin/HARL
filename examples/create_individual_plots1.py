import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('/project/LRGO-main/run_example/batch_experiment_results_averaged_partial.csv')

# Rename columns to match the script's expectations
df1 = df1.rename(columns={ 'ta_name': 'assignment', 'ag_num': 'num_agents', 'tar_num': 'goal_num', 'task_completed_rate': 'SR_avg', 'steps': 'MaxStep' })

# Normalize SR_avg to 0-1 range
df1['SR_avg'] = df1['SR_avg'] / 100.0

df1['num_agents'] = pd.to_numeric(df1['num_agents'])
df1['goal_num'] = pd.to_numeric(df1['goal_num'])
df1['SR_avg'] = pd.to_numeric(df1['SR_avg'])
df1['MaxStep'] = pd.to_numeric(df1['MaxStep'])

# Replace 9999 with NaN for success_episode_step_avg (indicating no success)
df1['MaxStep'] = df1['MaxStep'].replace(9999, np.nan)

# Colors for different method combinations
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '<']
# labellist ={'AMARLp':'AMARL+HCSPPO','DisPnp':'DisPn+HCSPPO','GoalSet':'End2EndDRL','lapjv':'Lapjv+HCSPPO','EAGAT1':'EATMP (Ours)'}
labellist ={'lrca': 'LRCA', 'dsta': 'DSTA', 'lsta': 'LSTA', 'cbba': 'CBBA'}

# Plot 1: agent num = 10, SR vary with goal num
plt.figure(figsize=(8, 6))
data_1 = df1[df1['num_agents'] == 10]
grouped_1 = data_1.groupby(['goal_num', 'assignment',])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_1[(grouped_1['assignment'] == assignment)]
    if len(subset) > 0:
        # print(subset['goal_num'], subset['SR_avg'])
        plt.plot(subset['goal_num'].to_numpy(), subset['SR_avg'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Goal Number $', fontsize=16)
plt.ylabel('Success Rate ($)', fontsize=16)
plt.title('=10$: Success Rate vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.ylim(0, 1.05)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot1_agents5_SR_vs_goals.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: agent num = 10, success_episode_step_avg vary with goal num
plt.figure(figsize=(8, 6))
data_2 = df1[(df1['num_agents'] == 10) & (~df1['MaxStep'].isna())]
grouped_2 = data_2.groupby(['goal_num', 'assignment', ])['MaxStep'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_2[(grouped_2['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['goal_num'].to_numpy(), subset['MaxStep'].to_numpy(), 
                       marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1


plt.xlabel('Goal Number $',  fontsize=16)
plt.ylabel('Episode Steps ({avg}$)',  fontsize=16)
plt.title('=10$: Episode Steps vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot2_agents5_steps_vs_goals.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: goal num = 100, SR vary with agent num
plt.figure(figsize=(8, 6))
data_3 = df1[df1['goal_num'] == 100]
grouped_3 = data_3.groupby(['num_agents', 'assignment',])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_3[(grouped_3['assignment'] == assignment) ]
    if len(subset) > 0:
        plt.plot(subset['num_agents'].to_numpy(), subset['SR_avg'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Number of Agents $',  fontsize=16)
plt.ylabel('Success Rate ($)',  fontsize=16)
plt.title('=100$: Success Rate vs Number of Agents',  fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.ylim(0, 1.05)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot3_goals100_SR_vs_agents.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: goal num = 100, success_episode_step_avg vary with agent num
plt.figure(figsize=(8, 6))
data_4 = df1[(df1['goal_num'] == 100) & (~df1['MaxStep'].isna())]
grouped_4 = data_4.groupby(['num_agents', 'assignment',])['MaxStep'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_4[(grouped_4['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['num_agents'].to_numpy(), subset['MaxStep'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Number of Agents $',  fontsize=16)
plt.ylabel('Episode Steps ({avg}$)',  fontsize=16)
plt.title('=100$: Episode Steps vs Number of Agents',  fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot4_goals100_steps_vs_agents.png', dpi=300, bbox_inches='tight')
plt.close()

print('Individual plots saved!')
print('Files created:')
print('1. plot1_agents5_SR_vs_goals.png')
print('2. plot2_agents5_steps_vs_goals.png') 
print('3. plot3_goals100_SR_vs_agents.png')
print('4. plot4_goals100_steps_vs_agents.png')
