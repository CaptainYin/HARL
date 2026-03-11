from tbparse import SummaryReader
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


import glob

# read in all events under the specified logs directory and merge
log_dir4v4hpn = "/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN2H_Cube_4v4/seed-00001-2025-06-11-10-54-48/logs"
log_dir4v4mlp = "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_4v4_randompos1/seed-00001-2025-05-14-15-43-07/logs"
log_dir10v10hpn = "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOHPN_Cube_10v10_randompos1/seed-00001-2025-05-14-15-40-08/logs"
log_dir10v10mlp = "/project/HARL-main/examples/results/lasercar/Cube/mappo/MAPPOMLP_Cube_10v10_randompos1/seed-00001-2025-05-14-18-06-45/logs"

log_dir4v4_4H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_4v4/seed-00001-2025-06-10-13-46-12/logs"
log_dir4v4_8H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_4v4/seed-00001-2025-06-10-13-50-05/logs"
log_dir4v4_16H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_4v4/seed-00001-2025-06-10-13-52-25/logs"
log_dir10v10_4H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN4H_Cube_10v10/seed-00001-2025-06-11-01-41-34/logs"
log_dir10v10_8H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN8H_Cube_10v10/seed-00001-2025-06-11-01-42-25/logs"
log_dir10v10_16H="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H_Cube_10v10/seed-00001-2025-06-11-01-42-43/logs"

dir4v416H128="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H128_Cube_4v4/seed-00001-2025-06-11-11-11-20/logs"
dir4v4HPN16H32="/project/HARL-main/examples/results/lasercar/Cube/mappo/HPN16H32_Cube_4v4/seed-00001-2025-06-11-11-10-41/logs"
run_list={"MLP-MAPPO_4v4":log_dir4v4mlp,"HCN-MAPPO_4v4":log_dir4v4hpn,"4v4_4H":log_dir4v4_4H, "4v4_8H":log_dir4v4_8H,"4v4_16H":log_dir4v4_16H,"4v4_16H32":dir4v4HPN16H32,"4v4_16H128":dir4v416H128,"MLP-MAPPO_10v10":log_dir10v10mlp,"HCN-MAPPO_10v10":log_dir10v10hpn,"10v10_4H":log_dir10v10_4H, "10v10_8H":log_dir10v10_8H,"10v10_16H":log_dir10v10_16H}
labels=["MLP-MAPPO_4v4","HCN-MAPPO_4v4","MLP-MAPPO_10v10","HCN-MAPPO_10v10"]
# "4v4_4H", "4v4_8H","4v4_16H",,"10v10_4H", "10v10_8H","10v10_16H"
def read_events(log_dir): 
    print(f"Reading events from {log_dir}")
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)
    df_list = []
    for ev in event_files:
        tmp = SummaryReader(ev).scalars.copy()
        tmp['filename'] = ev
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    df_wide = df.pivot(index='step', columns='tag', values='value')
    return df_wide

key_list=[ 'car_succ_rate',
        'critic/value_loss', 
        'eval_average_episode_rewards',
       'train_episode_rewards']#'car_collision_rate','car_timeout_rate', 'critic/critic_grad_norm','critic/value_preds','obs_collision_rate','critic/average_step_rewards','episode_succ_rate','eval_max_episode_rewards', 
# df_wide4v4hpn = read_events(log_dir4v4hpn)
# df_wide4v4mlp = read_events(log_dir4v4mlp)
# df_wide10v10hpn = read_events(log_dir10v10hpn)
# df_wide10v10mlp = read_events(log_dir10v10mlp)
df_list= [read_events(run_list[label]) for label in labels]
# orig_key = 'train_episode_rewards'
for orig_key in key_list:
    # plot agent0/actor_grad_norm from both datasets
    key = orig_key  # use for indexing into df_wide
    # sanitize for labels and filenames
    safe_key = orig_key.replace('/', '_')

    # series_4v4hpn = df_wide4v4hpn[key]
    # series_4v4mlp = df_wide4v4mlp[key]
    # series_10v10hpn = df_wide10v10hpn[key]
    # series_10v10mlp = df_wide10v10mlp[key]
    series= [df[key] for df in df_list]
    # apply rolling smoothing (e.g., window size 100)
    window = 20 if orig_key in ['car_succ_rate', 'episode_succ_rate','eval_average_episode_rewards','eval_max_episode_rewards'] else 100
    # series_4v4hpn = series_4v4hpn.rolling(window=window, min_periods=1, center=True).mean()
    # series_4v4mlp = series_4v4mlp.rolling(window=window, min_periods=1, center=True).mean()
    # series_10v10hpn = series_10v10hpn.rolling(window=window, min_periods=1, center=True).mean()
    # series_10v10mlp = series_10v10mlp.rolling(window=window, min_periods=1, center=True).mean()
    
    # Remove NaN values before calculating mean and std
    series_clean = [s.dropna() for s in series]
    
    # Calculate mean and std for each series (after removing NaN)
    series_mean = [s.rolling(window=window, min_periods=1, center=True).mean() for s in series_clean]
    series_std = [s.rolling(window=window, min_periods=1, center=True).std() for s in series_clean]

    fontsize=18
    markersize=1
    linewidth=1

    fig, ax = plt.subplots(figsize=(10,8),dpi=200)
    
    # Define colors for each series
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot with std shading
    for i, lab in enumerate(labels):
        color = colors[i % len(colors)]
        mean_series = series_mean[i]
        std_series = series_std[i]
        
        # Plot the mean line
        mean_series.plot(ax=ax, label=lab, marker='o', markersize=markersize, 
                        linewidth=linewidth, color=color, grid=True)
        
        # Add std shading (fill_between)
        x_values = mean_series.index
        y_mean = mean_series.values
        y_std = 0.5*std_series.values
        
        # Remove NaN values for fill_between
        mask = ~(pd.isna(y_mean) | pd.isna(y_std))
        if mask.any():
            ax.fill_between(x_values[mask], 
                           (y_mean - y_std)[mask], 
                           (y_mean + y_std)[mask], 
                           alpha=0.2, color=color)
    ax.set_xlabel('Step',fontsize=fontsize)

    if safe_key == 'car_succ_rate':
        ax.set_ylabel('Success Rate $R_s$',fontsize=fontsize)
    elif safe_key == 'critic_value_loss':
        ax.set_ylabel('Critic Value Loss',fontsize=fontsize)
    else:
        ax.set_ylabel(safe_key,fontsize=fontsize)
    plt.xticks(fontsize=fontsize,color='#000000')
    plt.yticks(fontsize=fontsize,color='#000000')
    # ax.set_title(f'{safe_key}: HPN vs MLP',fontsize=fontsize)
    ax.legend(loc='best',frameon=False,fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'fig/7_{safe_key}.png', dpi=300,bbox_inches ='tight')