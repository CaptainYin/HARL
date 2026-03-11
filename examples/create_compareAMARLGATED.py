import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from plot_analysis import load_and_cast
# df = pd.read_csv('/project/HARL-main/examples/render5.csv', header=None)

# Define column names based on the provided information
# columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
        #    'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg',]
columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
    'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg','SR_array','MRTA_count_array','Max_Step_array','Obs_CR_array','Car_CR_array','TimeoutR_array',]
# df.columns = columns
print("Loading data...")
df = load_and_cast('/project/HARL-main/examples/render_AMARLEAGAT_rb.csv', column_names=columns)
for agent_num in sorted(df['num_agents'].unique()):
    df_agent = df.loc[df['num_agents'] == agent_num]
    for goal_num in sorted(df_agent['goal_num'].unique()):
        sub_df = df_agent.loc[df_agent['goal_num'] == goal_num]
        print(f"Agents: {agent_num}, Goals: {goal_num}, Entries: {len(sub_df)}")
        success=None
        # sub_agents = df.loc[df['num_agents'] == agent_num]
        # sub_agents = sub_df
        for assgn in sub_df['assignment'].unique():
            if assgn in ['lapjv','GoalSet','DisPn']:
# DisPn GoalSet AMARL lapjv EAGAT
                continue
            if success is None:
                # print(df[df['num_agents']==10][df['goal_num']==20][df['assignment']==assgn]['SR_array'].values[0])
                success=sub_df.loc[sub_df['assignment'] == assgn, 'SR_array'].values[0]==1
                # if not success.any():
                #     continue
                
            else:
                # print((df[df['num_agents']==10][df['goal_num']==20][df['assignment']==assgn]['SR_array'].values[0]==1).any())
                success=success * sub_df.loc[sub_df['assignment'] == assgn, 'SR_array'].values[0]==1
                # print(success)
        print("Final success over all methods:",success.any())
        for assgn in ['EAGAT','AMARL']:
            # 'MRTA_count_array','Max_Step_array','Obs_CR_array','Car_CR_array'
            sub_assgn = sub_df.loc[sub_df['assignment'] == assgn]
            if len(sub_assgn['Car_CR_array'].values)==0:
                continue
            Car_CR_array = sub_assgn['Car_CR_array'].values[0]
            # Car_CR_array = sub_df.loc[sub_df['assignment'] == assgn, 'Car_CR_array'].values[0]
            # MRTA_count_array = sub_df.loc[sub_df['assignment'] == assgn, 'MRTA_count_array'].values[0]
            MRTA_count_array = sub_assgn['MRTA_count_array'].values[0]
            # Max_Step_array = sub_df.loc[sub_df['assignment'] == assgn, 'Max_Step_array'].values[0]
            Max_Step_array = sub_assgn['Max_Step_array'].values[0]
            # Obs_CR_array = sub_df.loc[sub_df['assignment'] == assgn, 'Obs_CR_array'].values[0]
            Obs_CR_array = sub_assgn['Obs_CR_array'].values[0]
            # sub_assgn['Car_CR_valid'] = Car_CR_array[success].mean() if success.any() else np.nan
            df.loc[(df['num_agents'] == agent_num) & (df['goal_num'] == goal_num) & (df['assignment'] == assgn), 'Car_CR_valid'] = Car_CR_array[success].mean() if success.any() else np.nan
            df.loc[(df['num_agents'] == agent_num) & (df['goal_num'] == goal_num) & (df['assignment'] == assgn), 'Obs_CR_valid'] = Obs_CR_array[success].mean() if success.any() else np.nan
            df.loc[(df['num_agents'] == agent_num) & (df['goal_num'] == goal_num) & (df['assignment'] == assgn), 'Max_Step_valid'] = Max_Step_array[success].mean() if success.any() else np.nan
            df.loc[(df['num_agents'] == agent_num) & (df['goal_num'] == goal_num) & (df['assignment'] == assgn), 'MRTA_count_valid'] = MRTA_count_array[success].mean() if success.any() else np.nan
            
df['num_agents'] = pd.to_numeric(df['num_agents'])
df['goal_num'] = pd.to_numeric(df['goal_num'])
df['SR_avg'] = pd.to_numeric(df['SR_avg'])
df['success_episode_step_avg'] = pd.to_numeric(df['success_episode_step_avg'])
df['MaxStep'] = pd.to_numeric(df['MaxStep'])
df['SuccessCarStep'] = pd.to_numeric(df['SuccessCarStep'])
# df['Obs_CR_valid'] = pd.to_numeric(df['Obs_CR_valid'])
# df['Car_CR_valid'] = pd.to_numeric(df['Car_CR_valid'])
df['CR']= df['Obs_CR_valid']+df['Car_CR_valid']
# print(df['Obs_CR_valid'])
# exit()
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
data_1 = df[(df['num_agents'] == 10) & (~df['CR'].isna())]
grouped_1 = data_1.groupby(['goal_num', 'assignment',])['CR'].mean().reset_index()
color_idx = 0
for assignment in labellist.keys():
    subset = grouped_1[(grouped_1['assignment'] == assignment)]
    if len(subset) > 0:
        # print(subset['goal_num'], subset['SR_avg'])
        plt.plot(subset['goal_num'].to_numpy(), subset['CR'].to_numpy(), 
                        marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Goal Number $N$', fontsize=16)
plt.ylabel('Collision Rate ($R_c$)', fontsize=16)
plt.title('$M=10$: Collision Rate vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
# plt.ylim(0, 1.05)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot_Collision_Rate.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: agent num = 5, success_episode_step_avg vary with goal num
plt.figure(figsize=(8, 6))
data_2 = df[(df['num_agents'] == 10) & (~df['Max_Step_valid'].isna())]
grouped_2 = data_2.groupby(['goal_num', 'assignment', ])['Max_Step_valid'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_2[(grouped_2['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['goal_num'].to_numpy(), subset['Max_Step_valid'].to_numpy(), 
                       marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Goal Number $N$',  fontsize=16)
plt.ylabel('Episode Steps ($T_{avg}$)',  fontsize=16)
plt.title('$M=10$: Episode Steps vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot_Episode_Steps.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: agent num = 5, success_episode_step_avg vary with goal num
plt.figure(figsize=(8, 6))
data_2 = df[(df['num_agents'] == 10) & (~df['MRTA_count_valid'].isna())]
grouped_2 = data_2.groupby(['goal_num', 'assignment', ])['MRTA_count_valid'].mean().reset_index()

color_idx = 0
for assignment in labellist.keys():
    subset = grouped_2[(grouped_2['assignment'] == assignment)]
    if len(subset) > 0:
        plt.plot(subset['goal_num'].to_numpy(), subset['MRTA_count_valid'].to_numpy(), 
                       marker=markers[color_idx % len(markers)], color=colors[color_idx % len(colors)], label=f'{labellist[assignment]}', linewidth=2.5, markersize=8)
        color_idx += 1

plt.xlabel('Goal Number $N$',  fontsize=16)
plt.ylabel('Allocation count',  fontsize=16)
plt.title('$M=10$: Allocation count vs Goal Number', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('/project/HARL-main/examples/plot_Allocation_count.png', dpi=300, bbox_inches='tight')
plt.close()
print("Individual plots saved!")
print("Files created:")
print("1. plot1_agents5_SR_vs_goals.png")
print("2. plot2_agents5_steps_vs_goals.png") 
