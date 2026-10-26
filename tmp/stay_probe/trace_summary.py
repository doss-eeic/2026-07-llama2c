"""Summarize separately instrumented worker runs (not timing trials)."""
import csv
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = []
for trial in sorted(root.glob('trace_*')):
    workers = {}
    for line in (trial / 'stderr.log').read_text().splitlines():
        if not line.startswith('probe,'):
            continue
        _, cpu, stage, count, dispatch, start, finish, cpu_ns = line.split(',')
        values = (int(count), (int(start)-int(dispatch))/1e6,
                  (int(finish)-int(start))/1e6, int(cpu_ns)/1e6)
        workers.setdefault((cpu, stage), []).append(values)
    for (cpu, stage), values in workers.items():
        values = values[1:]
        row = {'config': trial.name.removeprefix('trace_'), 'cpu': int(cpu), 'stage': stage,
               'jobs': len(values)}
        for i, label in enumerate(('rows', 'wake_ms', 'work_wall_ms', 'work_cpu_ms')):
            row[label + ':median'] = statistics.median(v[i] for v in values)
        rows.append(row)
with (root / 'workers.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    if row['stage'] == 'output':
        print(row)
