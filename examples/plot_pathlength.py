import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV
columns = ['scenario', 'algorithm', 'modelname', 'agent_num', 'goal_num', 'gurobi_length', 'model_length', 'model_gurobi_ratio']
df = pd.read_csv('/project/MinMax-MTSP-master/partition/path.csv', header=None, names=columns)

# Convert numeric columns
df['agent_num'] = pd.to_numeric(df['agent_num'])
df['goal_num'] = pd.to_numeric(df['goal_num'])
df['gurobi_length'] = pd.to_numeric(df['gurobi_length'])
df['model_length'] = pd.to_numeric(df['model_length'])
df['model_gurobi_ratio'] = pd.to_numeric(df['model_gurobi_ratio'])

plt.style.use('default')
sns.set_palette("husl")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model/Gurobi Ratio Analysis', fontsize=16, fontweight='bold')

# Figure 1: agent_num=5, model_gurobi_ratio vs goal_num
data1 = df[df['agent_num'] == 5]
grouped_1 = data1.groupby(['goal_num', 'algorithm',])['model_length'].mean().reset_index()

for algo in grouped_1['algorithm'].unique():
    subset = grouped_1[grouped_1['algorithm'] == algo]
    # print(subset['goal_num'], subset['model_gurobi_ratio'])
    axes[0].plot(subset['goal_num'].to_numpy(), subset['model_length'].to_numpy(), marker='o', label=algo)
grouped_1 = data1.groupby(['goal_num', 'algorithm',])['gurobi_length'].mean().reset_index()
subset = grouped_1[grouped_1['algorithm'] == algo]
axes[0].plot(subset['goal_num'].to_numpy(), subset['gurobi_length'].to_numpy(), marker='o', label='Gurobi', linestyle='--', color='black')
axes[0].set_xlabel('Goal Number')
axes[0].set_ylabel('Model/Gurobi Ratio')
axes[0].set_title('Fixed agent_num=5')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Figure 2: goal_num=100, model_gurobi_ratio vs agent_num
data2 = df[df['goal_num'] == 100]
grouped_1 = data2.groupby(['agent_num', 'algorithm',])['model_length'].mean().reset_index()
for algo in grouped_1['algorithm'].unique():
    subset = grouped_1[grouped_1['algorithm'] == algo]
    axes[1].plot(subset['agent_num'].to_numpy(), subset['model_length'].to_numpy(), marker='s', label=algo)
grouped_1 = data2.groupby(['agent_num', 'algorithm',])['gurobi_length'].mean().reset_index()
subset = grouped_1[grouped_1['algorithm'] == algo]
axes[1].plot(subset['agent_num'].to_numpy(), subset['gurobi_length'].to_numpy(), marker='o', label='Gurobi', linestyle='--', color='black')
axes[1].set_xlabel('Agent Number')
axes[1].set_ylabel('Model/Gurobi Ratio')
axes[1].set_title('Fixed goal_num=100')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('partition_ratio_analysis.png', dpi=300)
print("Plot saved to partition_ratio_analysis.png")