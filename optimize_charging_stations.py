import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

# Config
INPUT_DIR = "berlin_friedrichshain_simplified"
NODES_FILE = f"{INPUT_DIR}/nodes.csv"
LINKS_FILE = f"{INPUT_DIR}/links.csv"
SETTINGS_OUT = f"{INPUT_DIR}/settings.json"
TARGET_COUNT = 20
DEFAULT_PARAMS = [0.5, 2.0]

print("Loading network...")
df_nodes = pd.read_csv(NODES_FILE)
df_links = pd.read_csv(LINKS_FILE)

# 1. Build Connectivity Graph (Undirected/Neighbor-based)
# We want to count unique neighbors to identify "dead ends" or "boundary nodes".
adj = defaultdict(set)
for _, row in df_links.iterrows():
    u, v = row['start'], row['end']
    if u != v:
        adj[u].add(v)
        adj[v].add(u) # Treat as undirected for neighbor counting

# 2. Filter Candidates
# Criteria: Node must have at least 2 unique neighbors (Degree >= 2)
# Ideally >= 3 for intersections, but let's start with >= 2 to avoid pure dead-ends.
candidates = []
rejected = []

for _, row in df_nodes.iterrows():
    nid = row['name']
    degree = len(adj[nid])
    
    if degree >= 2:
        candidates.append(row)
    else:
        rejected.append(nid)

df_candidates = pd.DataFrame(candidates)

print(f"Total Nodes: {len(df_nodes)}")
print(f"Candidates (Degree >= 2): {len(df_candidates)}")
print(f"Rejected (Degree < 2): {len(rejected)}")

# If we have fewer candidates than target, we must fallback (unlikely here)
if len(df_candidates) < TARGET_COUNT:
    print("WARNING: Not enough candidates with Degree >= 2. Relaxing constraint.")
    df_candidates = df_nodes # Fallback to all

# 3. Furthest Point Sampling on Candidates
# Extract coordinates of candidates
cand_coords = df_candidates[['x', 'y']].values
cand_ids = df_candidates['name'].values

# Start with the node closest to the geometric center of the candidates (Centrality)
# Or just a random one. Let's pick the one closest to mean position to start "in the middle"
# actually FPS usually starts with a boundary or random. 
# If we start "center", FPS will pick boundaries next.
# If we start "random", it might be boundary.
# Let's start with the candidate that has the HIGHEST degree (Major hub) to ensure at least one big hub is picked.
# Then FPS will spread out.

cand_degrees = [len(adj[nid]) for nid in cand_ids]
start_idx = np.argmax(cand_degrees) # Index in cand_ids/cand_coords

selected_indices = [start_idx]
dist_matrix = squareform(pdist(cand_coords))

# Initialize distances
min_dists = dist_matrix[start_idx].copy()

while len(selected_indices) < TARGET_COUNT:
    current_min_dists = min_dists.copy()
    current_min_dists[selected_indices] = -1
    
    next_idx = np.argmax(current_min_dists)
    selected_indices.append(next_idx)
    
    new_dists = dist_matrix[next_idx]
    min_dists = np.minimum(min_dists, new_dists)

selected_node_ids = cand_ids[selected_indices]

# Verify degrees of selected nodes
print("\nSelected Charging Nodes Stats:")
print(f"{ 'Node ID':<10} | { 'Degree':<10}")
print("-" * 25)
for nid in selected_node_ids:
    print(f"{nid:<10} | {len(adj[nid]):<10}")

# 4. Update Settings
try:
    with open(SETTINGS_OUT, 'r') as f:
        settings = json.load(f)
except FileNotFoundError:
    settings = {}

charging_nodes_dict = {}
for nid in selected_node_ids:
    charging_nodes_dict[str(int(nid))] = DEFAULT_PARAMS

settings["charging_nodes"] = charging_nodes_dict

with open(SETTINGS_OUT, 'w') as f:
    json.dump(settings, f, indent=4)

print(f"\nSettings updated in {SETTINGS_OUT}")

# 5. (Optional) Quick Visualization Check
# We can reuse the verify script if needed, or just trust the stats.
