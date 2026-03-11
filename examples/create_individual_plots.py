import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/project/HARL-main/examples/render4.csv', header=None)

# Define column names based on the provided information
# columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
        #    'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg',]
columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
    'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg','SR_array','MRTA_count_array','Max_Step_array','Obs_CR_array','Car_CR_array','TimeoutR_array',]
df.columns = columns

df['num_agents'] = pd.to_numeric(df['num_agents'])
df['goal_num'] = pd.to_numeric(df['goal_num'])
df['SR_avg'] = pd.to_numeric(df['SR_avg'])
df['success_episode_step_avg'] = pd.to_numeric(df['success_episode_step_avg'])
df['MaxStep'] = pd.to_numeric(df['MaxStep'])
df['SuccessCarStep'] = pd.to_numeric(df['SuccessCarStep'])

# Replace 9999 with NaN for success_episode_step_avg (indicating no success)
df['success_episode_step_avg'] = df['success_episode_step_avg'].replace(9999, np.nan)
df['SuccessCarStep'] = df['SuccessCarStep'].replace(9999, np.nan)
df['MaxStep'] = df['MaxStep'].replace(9999, np.nan)
# Colors for different method combinations
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '<']
# labellist ={'AMARLp':'AMARL+HCSPPO','DisPnp':'DisPn+HCSPPO','GoalSet':'End2EndDRL','lapjv':'Lapjv+HCSPPO','EAGAT1':'EATMP (Ours)'}
labellist ={'AMARL':'AMARL+SPOMC','DisPn':'DisPn+SPOMC','EAGAT':'GASP (Ours)','GoalSet':'End2EndDRL','lapjv':'Lapjv+SPOMC'}
# Plot 1: agent num = 5, SR vary with goal num
plt.figure(figsize=(8, 6))
data_1 = df[df['num_agents'] == 10]
grouped_1 = data_1.groupby(['goal_num', 'assignment',])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_1[(grouped_1['assignment'] == assignment)]
    if len(subset) > 0:
        # print(subset['goal_num'], subset['SR_avg'])
        plt.plot(subset['goal_num'].to_numpy(), subset['SR_avg'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Goal Number $N$', fontsize=16)
plt.ylabel('Success Rate ($R_s$)', fontsize=16)
plt.title('$M=10$: Success Rate vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.ylim(0, 1.05)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot1_agents5_SR_vs_goals.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: agent num = 5, success_episode_step_avg vary with goal num
plt.figure(figsize=(8, 6))
data_2 = df[(df['num_agents'] == 5) & (~df['MaxStep'].isna())]
grouped_2 = data_2.groupby(['goal_num', 'assignment', ])['MaxStep'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_2[(grouped_2['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['goal_num'].to_numpy(), subset['MaxStep'].to_numpy(), 
                       marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1


plt.xlabel('Goal Number $N$',  fontsize=16)
plt.ylabel('Episode Steps ($S_{avg}$)',  fontsize=16)
plt.title('$M=5$: Episode Steps vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot2_agents5_steps_vs_goals.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: goal num = 100, SR vary with agent num
plt.figure(figsize=(8, 6))
data_3 = df[df['goal_num'] == 100]
grouped_3 = data_3.groupby(['num_agents', 'assignment',])['SR_avg'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_3[(grouped_3['assignment'] == assignment) ]
    if len(subset) > 0:
        plt.plot(subset['num_agents'].to_numpy(), subset['SR_avg'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Number of Agents $M$',  fontsize=16)
plt.ylabel('Success Rate ($R_s$)',  fontsize=16)
plt.title('$N=100$: Success Rate vs Number of Agents',  fontsize=16)
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
data_4 = df[(df['goal_num'] == 100) & (~df['MaxStep'].isna())]
grouped_4 = data_4.groupby(['num_agents', 'assignment',])['MaxStep'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_4[(grouped_4['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['num_agents'].to_numpy(), subset['MaxStep'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Number of Agents $M$',  fontsize=16)
plt.ylabel('Episode Steps ($S_{avg}$)',  fontsize=16)
plt.title('$N=100$: Episode Steps vs Number of Agents',  fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot4_goals100_steps_vs_agents.png', dpi=300, bbox_inches='tight')
plt.close()

print("Individual plots saved!")
print("Files created:")
print("1. plot1_agents5_SR_vs_goals.png")
print("2. plot2_agents5_steps_vs_goals.png") 
print("3. plot3_goals100_SR_vs_agents.png")
print("4. plot4_goals100_steps_vs_agents.png")
