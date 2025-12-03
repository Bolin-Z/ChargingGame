import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Load data
data_dir = "berlin_friedrichshain"
nodes_path = f"{data_dir}/{data_dir}_nodes.csv"
links_path = f"{data_dir}/{data_dir}_links.csv"

df_nodes = pd.read_csv(nodes_path)
df_links = pd.read_csv(links_path)

# Coordinates
coords = df_nodes[['x', 'y']].values
node_names = df_nodes['name'].values
n = len(df_nodes)

# 1. Spatial Clustering (50m threshold)
dist_matrix = squareform(pdist(coords))
threshold = 50.0

# Simple transitive clustering using Union-Find logic
parent = list(range(n))

def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_j] = root_i

# Apply distance threshold
for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] < threshold:
            union(i, j)

# Group nodes
clusters = {}
for i in range(n):
    root = find(i)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(i)

# Calculate new nodes (Centroids)
new_nodes = []
old_to_new_map = {} # old_name -> new_id

for new_id, (root, members) in enumerate(clusters.items(), 1):
    # Calculate centroid
    member_coords = coords[members]
    centroid = member_coords.mean(axis=0)
    
    new_nodes.append({
        'id': new_id,
        'x': centroid[0],
        'y': centroid[1],
        'member_count': len(members)
    })
    
    for member_idx in members:
        old_name = node_names[member_idx]
        old_to_new_map[old_name] = new_id

df_new_nodes = pd.DataFrame(new_nodes)

# 2. Link Analysis
new_links = []
min_len_old = df_links['length'].min()
lengths_new = []

# Helper to calculate distance
def calc_dist(n1, n2):
    p1 = df_new_nodes[df_new_nodes['id'] == n1].iloc[0]
    p2 = df_new_nodes[df_new_nodes['id'] == n2].iloc[0]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

for _, row in df_links.iterrows():
    start_old = row['start']
    end_old = row['end']
    
    if start_old not in old_to_new_map or end_old not in old_to_new_map:
        continue # Should not happen if data consistent

    new_start = old_to_new_map[start_old]
    new_end = old_to_new_map[end_old]
    
    if new_start != new_end:
        dist = calc_dist(new_start, new_end)
        lengths_new.append(dist)

min_len_new = min(lengths_new) if lengths_new else 0

# 3. Topology Check
# Construct simple graph for degree
from collections import defaultdict
degree = defaultdict(int)
for _, row in df_links.iterrows():
    s = old_to_new_map.get(row['start'])
    e = old_to_new_map.get(row['end'])
    if s and e and s != e:
        degree[s] += 1
        # degree[e] += 1 # Out-degree usually or total degree? 
        # Usually simplified networks are analyzed by total incident edges or just nodes.
        # Let's just count unique connections for "degree" approximation in this context
        
# Let's strictly follow the "Average Degree" usually means avg number of edges per node
# But here we are just doing a quick check.

print("=== Scheme Evaluation Report (50m Threshold) ===")
print(f"Original Nodes: {n}")
print(f"New Nodes: {len(df_new_nodes)}")
print(f"Reduction: {100 * (n - len(df_new_nodes)) / n:.1f}%")
print(f"Original Min Length: {min_len_old:.1f}m")
print(f"New Min Length (inter-node): {min_len_new:.1f}m")
print(f"Max Cluster Size: {df_new_nodes['member_count'].max()}")
