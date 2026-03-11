import csv
import re
from collections import defaultdict
from pathlib import Path

CSV_FILE = Path(__file__).parent / 'rendermix.csv'
CA_THRESHOLD = 0.35
UNMAZE_THRESHOLD = 0.915

STEP_RE = re.compile(r'actor_agent0_(\d+)\.pt')

# Data structure: step -> scenario -> (success, full_row)
results = defaultdict(dict)

with CSV_FILE.open('r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        # Expected row layout:
        # 0 algo, 1 scenario, 2 assign, 3 planner, 4 n1, 5 n2, 6 path, 7 success, 8 maybe collision, 9 metric
        if len(row) < 8:
            continue
        scenario = row[1].strip()
        model_path = row[6].strip()
        try:
            success = float(row[7])
        except ValueError:
            continue
        m = STEP_RE.search(model_path)
        if not m:
            continue
        step = int(m.group(1))
        # Keep the highest success per scenario+step if duplicates appear
        prev = results[step].get(scenario)
        if prev is None or success > prev[0]:
            results[step][scenario] = (success, row)

qualified = []
for step, scenemap in results.items():
    ca = scenemap.get('CA')
    unmaze = scenemap.get('Unmaze')
    cube = scenemap.get('Cube')
    warehouse = scenemap.get('Warehouse')
    if ca[0] > 0.345  and warehouse[0] > 0.925:
        qualified.append({
            'step': step,
            'ca_success': ca[0],
            'unmaze_success': unmaze[0],
            'cube_success': cube[0],
            'warehouse_success': warehouse[0],
            'model_path': scenemap['CA'][1][6]  # same path across scenarios
        })

qualified.sort(key=lambda x: x['step'])

print(f'Found {len(qualified)} model steps meeting thresholds: CA > {CA_THRESHOLD}, Unmaze > {UNMAZE_THRESHOLD}')
for q in qualified:
    print(f"step={q['step']}: CA={q['ca_success']:.4f}, Unmaze={q['unmaze_success']:.4f}, Cube={q['cube_success']:.4f}, Warehouse={q['warehouse_success']:.4f}, path={q['model_path']}")

# If you want the best (e.g., highest Unmaze then CA) you can uncomment below:
# if qualified:
#     best = max(qualified, key=lambda x: (x['unmaze_success'], x['ca_success']))
#     print('\nBest model based on (Unmaze, CA):')
#     print(best)
