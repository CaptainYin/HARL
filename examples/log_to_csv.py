import re
import csv
import os

def parse_log(log_path, csv_path):
    """
    Parse the onpolicybaserunner log and write metrics to CSV.
    """
    header = ['task','scenario','assignment','num_agents','goal_num','model_dir',
              'Success Rate','CR','success_episode_step_avg','Max_Step','success_car_step','Obs_CR','Car_CR','TimeoutR']
    with open(log_path, 'r') as logfile, open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        lines = logfile.readlines()
        n = len(lines)
        i = 0
        while i < n:
            line = lines[i]
            if 'task:' in line:
                # Parse task line
                # Extract content after 'task: '
                parts = line.split('task: ')[1].strip()
                # Split by comma, first element is task
                elems = [e.strip() for e in parts.split(',')]
                row = {k: '' for k in header}
                row['task'] = elems[0]
                # Parse remaining key/value pairs
                for e in elems[1:]:
                    if ' ' in e:
                        k, v = e.split(' ', 1)
                        row[k] = v
                # Next line contains metrics
                if i+1 < n:
                    metrics_line = lines[i+1]
                    # Extract after '] '
                    if '] ' in metrics_line:
                        metrics_str = metrics_line.split('] ', 1)[1].strip()
                    else:
                        metrics_str = metrics_line.strip()
                    metrics = [m.strip() for m in metrics_str.split(',')]
                    for m in metrics:
                        if ':' in m:
                            mk, mv = [x.strip() for x in m.split(':', 1)]
                            # Remove spaces in key for CSV header match
                            row_key = mk if mk in header else mk
                            row[row_key] = mv
                # Write row in header order
                writer.writerow([row.get(col, '') for col in header])
                i += 2
            else:
                i += 1

if __name__ == '__main__':
    # Adjust paths as needed
    log_file = os.path.join(os.path.dirname(__file__), 'onpolicybaserunner.log')
    csv_file = os.path.join(os.path.dirname(__file__), 'onpolicybaserunner_4V4MLP.csv')
    parse_log(log_file, csv_file)
    print(f"Converted {log_file} to {csv_file}")
