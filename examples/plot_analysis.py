import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import ast
# Read the CSV file
# render1-3.csv
# df = pd.read_csv('/project/HARL-main/examples/render3.csv', header=None)
csv_path = '/project/HARL-main/examples/render_10v20.csv'
# Define column names based on the provided information
columns = ['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
           'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg','SR_array','MRTA_count_array','Max_Step_array','Obs_CR_array','Car_CR_array','TimeoutR_array','Car_CR_valid','Obs_CR_valid','Max_Step_valid','MRTA_count_valid']

# df.columns = columns

def parse_possible_array(val):
    """
    Parse a string value that may contain an array-like representation such as:
      "[1.  1.  1.  1.  0.9]" or "[ 6 11 18 23 29]" or "[]" or "1.23"
    Returns:
      - numpy array of ints if all elements are integer-valued
      - numpy array of floats if floats present
      - numeric scalar (int/float) if input is a single numeric string
      - original string if nothing could be parsed
    """
    if pd.isna(val):
        return np.array([])

    if isinstance(val, (list, tuple, np.ndarray)):
        return np.array(val)

    s = str(val).strip()
    if s == "":
        return np.array([])

    # If it looks like a simple numeric scalar, try converting
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s):
        # integer or float
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        else:
            return int(s)

    # Remove surrounding quotes if present
    if (s[0] == s[-1]) and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    # Strip surrounding brackets/parentheses
    inner = re.sub(r'^[\[\(]+|[\]\)]+$', '', s).strip()
    if inner == "":
        return np.array([])

    # Replace commas with spaces, collapse multiple spaces to one
    inner_clean = inner.replace(',', ' ')
    inner_clean = re.sub(r'\s+', ' ', inner_clean).strip()

    # Attempt to parse with numpy.fromstring (handles space-separated and comma-separated numbers)
    try:
        arr = np.fromstring(inner_clean, sep=' ')
        if arr.size > 0:
            # If all are integer-valued, return ints
            if np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)):
                return arr.astype(int)
            return arr  # floats
    except Exception:
        pass

    # Fallback: try ast.literal_eval (handles python lists e.g. "[1,2,3]")
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (list, tuple)):
            arr = np.array(lit)
            # convert to int if possible
            if np.allclose(arr, np.round(arr)):
                return arr.astype(int)
            return arr
        # if it's a scalar literal
        if isinstance(lit, (int, float)):
            return lit
    except Exception:
        pass

    # Last resort: extract all numbers by regex
    nums = re.findall(r"[+-]?\d*\.\d+|[+-]?\d+", inner_clean)
    if nums:
        arr = np.array([float(x) for x in nums])
        if np.allclose(arr, np.round(arr)):
            return arr.astype(int)
        return arr

    # Could not parse -> return original string
    return s


def load_and_cast(csv_path, column_names=None, verbose=False):
    # Read CSV. Use engine='python' to be robust to weird separators,
    # but default should work for your example where arrays are space-separated inside brackets.
    df = pd.read_csv(csv_path, header=None, names=(column_names if column_names else cols), engine='python')

    # Columns to coerce to numeric scalars
    scalar_numeric_cols = ['SR_avg', 'Obs_CR_avg', 'Car_CR_avg', 'TimeoutRate',
                           'MaxStep', 'SuccessCarStep', 'success_episode_step_avg']

    for c in scalar_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Columns that should be parsed to arrays
    array_cols = ['SR_array', 'MRTA_count_array', 'Max_Step_array',
                  'Obs_CR_array', 'Car_CR_array', 'TimeoutR_array']

    for c in array_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_possible_array)

    if verbose:
        print("Loaded DataFrame with shape:", df.shape)
        print("Dtypes / sample values:")
        print(df.loc[0, scalar_numeric_cols + array_cols].to_dict())

    return df

if __name__ == "__main__":
    df = load_and_cast(csv_path, column_names=columns)

    # print(len(df[df['num_agents']==10]['SR_array']))
    # print(df[df['num_agents']==10][df['goal_num']==20]['assignment'].unique())

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
                # print(sub_df.loc[sub_df['assignment'] == assgn, 'Car_CR_valid'])
            
            # print(Car_CR_array[success],MRTA_count_array[success],Max_Step_array[success],Obs_CR_array[success])
        # print(assgn,df[df['num_agents']==10][df['goal_num']==20][df['assignment']==assgn]['SR_array'].values[0])
    # print(df[df['num_agents']==10][df['goal_num']==100][df['assignment']=='EAGAT']['Car_CR_valid'])
    # print(df['MRTA_count_valid'])
    # print(type(df['Car_CR_array']),df['Car_CR_array'],df['Obs_CR_array'],df['Max_Step_array'],df['MRTA_count_array'],df['TimeoutR_array'])

    df.to_csv( '/project/HARL-main/examples/render_10V20_processed.csv', index=False,columns=['task', 'scenario', 'assignment','assign_model', 'metric', 'num_agents', 'goal_num', 
            'model_dir', 'SR_avg', 'Obs_CR_avg','Car_CR_avg','TimeoutRate','MaxStep','SuccessCarStep', 'success_episode_step_avg','Car_CR_valid','Obs_CR_valid','Max_Step_valid','MRTA_count_valid'])
    exit()
    # Convert numeric columns
    df['num_agents'] = pd.to_numeric(df['num_agents'])
    df['goal_num'] = pd.to_numeric(df['goal_num'])
    df['SR_avg'] = pd.to_numeric(df['SR_avg'])
    df['success_episode_step_avg'] = pd.to_numeric(df['success_episode_step_avg'])
    df['MaxStep'] = pd.to_numeric(df['MaxStep'])
    df['SuccessCarStep'] = pd.to_numeric(df['SuccessCarStep'])
    df['Obs_CR_valid'] = pd.to_numeric(df['Obs_CR_valid'])
    df['Car_CR_valid'] = pd.to_numeric(df['Car_CR_valid'])
    df['CR']= df['Obs_CR_valid']+df['Car_CR_valid']
    # Replace 9999 with NaN for success_episode_step_avg (indicating no success)
    df['success_episode_step_avg'] = df['success_episode_step_avg'].replace(9999, np.nan)
    df['SuccessCarStep'] = df['SuccessCarStep'].replace(9999, np.nan)
    df['MaxStep'] = df['MaxStep'].replace(9999, np.nan)

    print("Data shape:", df.shape)
    print("\nData info:")
    print(df.info())
    print("\nUnique values:")
    print("Agent numbers:", sorted(df['num_agents'].unique()))
    print("Goal numbers:", sorted(df['goal_num'].unique()))
    print("Assignments:", df['assignment'].unique())
    print("Metrics:", df['metric'].unique())
    #'GoalSet':'End2EndDRL','lapjv':'Lapjv+HCSPPO','AMARL':'AMARL+HCSPPO','DisPn':'DisPn+HCSPPO','CMPNN4_13':'CMPGARL4_13+HCSPPO',,'CMPNNGAT':'CMPNNGAT+HCSPPO'
    # labellist ={'AMARLp':'AMARL+SPOMC','DisPnp':'DisPn+SPOMC','EAGAT1':'GASTAMP (Ours)','GoalSet':'End2EndDRL','lapjv':'Lapjv+SPOMC'}
    labellist ={'AMARL':'AMARL+SPOMC','DisPn':'DisPn+SPOMC','EAGAT':'GASTAMP (Ours)','GoalSet':'End2EndDRL','lapjv':'Lapjv+SPOMC'}
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Performance Analysis: Success Rate and Episode Steps', fontsize=16, fontweight='bold')

    # Plot 1: Fixed agent num = 5, SR vary with goal num
    print("\n=== Plot 1: Fixed agents=10, SR vs goal_num ===")
    data_1 = df[df['num_agents'] == 10]
    grouped_1 = data_1.groupby(['goal_num', 'assignment',])['SR_avg'].mean().reset_index()
    print(f"Data points for agents=5: {len(data_1)}")

    # Plot different combinations
    # for assignment in grouped_1['assignment'].unique():
    for assignment in labellist.keys():
        subset = grouped_1[(grouped_1['assignment'] == assignment)]
        if len(subset) > 0:
            # print(subset['goal_num'], subset['SR_avg'])
            axes[0,0].plot(subset['goal_num'].to_numpy(), subset['SR_avg'].to_numpy(), 
                            marker='o', label=f'{labellist[assignment]}', linewidth=2, markersize=6)

    axes[0,0].set_xlabel('Goal Number', fontweight='bold')
    axes[0,0].set_ylabel('Success Rate (SR)', fontweight='bold')
    axes[0,0].set_title('Fixed Agents=5: Success Rate vs Goal Number', fontweight='bold')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(left=0)
    axes[0,0].set_ylim(0, 1.05)

    # Plot 2: Fixed agent num = 5, success_episode_step_avg vary with goal num
    print("\n=== Plot 2: Fixed agents=5, MaxStep vs goal_num ===")
    data_2 = df[(df['num_agents'] == 10) & (~df['MRTA_count_valid'].isna())]
    grouped_2 = data_2.groupby(['goal_num', 'assignment', ])['MRTA_count_valid'].mean().reset_index()
    print(f"Data points for agents=5 (valid steps): {len(data_2)}")

    # for assignment in grouped_2['assignment'].unique():
    for assignment in labellist.keys():
        subset = grouped_2[(grouped_2['assignment'] == assignment)]
        if len(subset) > 0:
            axes[0,1].plot(subset['goal_num'].to_numpy(), subset['MRTA_count_valid'].to_numpy(), 
                            marker='s', label=f'{labellist[assignment]}', linewidth=2, markersize=6)

    axes[0,1].set_xlabel('Goal Number', fontweight='bold')
    axes[0,1].set_ylabel('Episode Steps (avg)', fontweight='bold')
    axes[0,1].set_title('Fixed Agents=5: Episode Steps vs Goal Number', fontweight='bold')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(left=0)

    # Plot 3: Fixed goal num = 100, SR vary with agent num
    print("\n=== Plot 3: Fixed goals=50, SR vs agent_num ===")
    data_3 = df[df['goal_num'] == 100]
    grouped_3 = data_3.groupby(['num_agents', 'assignment',])['SR_avg'].mean().reset_index()
    print(f"Data points for goals=100: {len(data_3)}")

    # for assignment in grouped_3['assignment'].unique():
    for assignment in labellist.keys():
        subset = grouped_3[(grouped_3['assignment'] == assignment) ]
        if len(subset) > 0:
            axes[1,0].plot(subset['num_agents'].to_numpy(), subset['SR_avg'].to_numpy(), 
                            marker='^', label=f'{labellist[assignment]}', linewidth=2, markersize=6)

    axes[1,0].set_xlabel('Number of Agents', fontweight='bold')
    axes[1,0].set_ylabel('Success Rate (SR)', fontweight='bold')
    axes[1,0].set_title('Fixed Goals=100: Success Rate vs Number of Agents', fontweight='bold')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(left=0)
    axes[1,0].set_ylim(0, 1.05)

    # Plot 4: Fixed goal num = 100, MaxStep vary with agent num
    print("\n=== Plot 4: Fixed goals=50, MaxStep vs agent_num ===")
    data_4 = df[(df['goal_num'] == 100) & (~df['MRTA_count_valid'].isna())]
    grouped_4 = data_4.groupby(['num_agents', 'assignment',])['MRTA_count_valid'].mean().reset_index()
    print(f"Data points for goals=100 (valid steps): {len(data_4)}")

    # for assignment in grouped_4['assignment'].unique():
    for assignment in labellist.keys():
        subset = grouped_4[(grouped_4['assignment'] == assignment)]
        if len(subset) > 0:
            axes[1,1].plot(subset['num_agents'].to_numpy(), subset['MRTA_count_valid'].to_numpy(), 
                            marker='d', label=f'{labellist[assignment]}', linewidth=2, markersize=6)

    axes[1,1].set_xlabel('Number of Agents', fontweight='bold')
    axes[1,1].set_ylabel('Episode Steps (avg)', fontweight='bold')
    axes[1,1].set_title('Fixed Goals=100: Episode Steps vs Number of Agents', fontweight='bold')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
