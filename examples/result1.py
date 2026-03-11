import pandas as pd

# 1. read the CSV
df = pd.read_csv("results_summary1001.csv")

# 2. make sure SR is numeric (sometimes it comes in as string)
df["SR"] = pd.to_numeric(df["SR"], errors="coerce")

def get_subtable(df,
                 model_name=None,
                 num_agents=None,
                 goal_num=None,
                 scenario=None,
                 sort_by="SR",
                 ascending=False):
    """
    Return a filtered & sorted sub-DataFrame.
    Any parameter set to None is ignored in the filter.
    """
    sub = df.copy()
    if model_name is not None:
        sub = sub[sub["model_name"] == model_name]
    if num_agents is not None:
        sub = sub[sub["num_agents"] == num_agents]
    if goal_num is not None:
        sub = sub[sub["goal_num"] == goal_num]
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario]

    return sub.sort_values(by=sort_by, ascending=ascending)

# 3. Example usage
#    a) a single sub-table
# st = get_subtable(df,
#                   model_name="A",
#                   num_agents=2,
#                   goal_num=3,
#                   scenario="foo")
# print(st)

#    b) write all scenarios for each (model_name, num_agents, goal_num) into one Excel workbook
import itertools  # ensure itertools is imported
from pandas import ExcelWriter

with ExcelWriter("subtables1.xlsx") as writer:
    # for each model/agents/goals group, write scenario blocks with two-row gaps
    for (m, na, gn), group in df.groupby(["model_name", "num_agents", "goal_num"]):
        if na!=gn:
            print(f"Skipping group with num_agents {na} != goal_num {gn}")
            continue
        sheet_name = f"{m}_{na}_{gn}"[:31]
        startrow = 0
        # iterate scenarios in ascending order
        for scenario, scen_group in group.groupby("scenario"):
            # sort within scenario by SR descending
            scen_sorted = scen_group.sort_values("SR", ascending=False)
            # include header only for first block
            scen_sorted.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=startrow,
                header=(startrow == 0)
            )
            # advance startrow by rows written plus 2 blank rows gap
            startrow += len(scen_sorted) + 2
        print(f"Wrote groups for sheet '{sheet_name}' with {len(group)} total rows across {group['scenario'].nunique()} scenarios")
print("All groups written to subtables1.xlsx")