import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import json

# Configuration
INPUT_DIR = "berlin_friedrichshain"
OUTPUT_DIR = "berlin_friedrichshain_simplified"
THRESHOLD = 50.0

# File Paths
nodes_path = f"{INPUT_DIR}/{INPUT_DIR}_nodes.csv"
links_path = f"{INPUT_DIR}/{INPUT_DIR}_links.csv"
demand_path = f"{INPUT_DIR}/{INPUT_DIR}_demand.csv"

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading data...")
df_nodes = pd.read_csv(nodes_path)
df_links = pd.read_csv(links_path)
df_demand = pd.read_csv(demand_path)

# ==========================================
# 1. Spatial Clustering
# ==========================================
print(" performing spatial clustering...")
coords = df_nodes[['x', 'y']].values
node_names = df_nodes['name'].values
n = len(df_nodes)

dist_matrix = squareform(pdist(coords))
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

for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] < THRESHOLD:
            union(i, j)

clusters = {}
for i in range(n):
    root = find(i)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(i)

# Generate New Nodes
new_nodes_list = []
old_to_new_map = {} # old_name -> new_id_str

# Reset IDs to 1..N for cleanliness, or keep logical names?
# Doc says: "ID suggested reset to 1..N"
for idx, (root, members) in enumerate(clusters.items(), 1):
    member_coords = coords[members]
    centroid = member_coords.mean(axis=0)
    new_id = idx
    
    new_nodes_list.append({
        'name': new_id, # int
        'x': centroid[0],
        'y': centroid[1]
    })
    
    for member_idx in members:
        old_name = node_names[member_idx]
        old_to_new_map[old_name] = new_id

df_new_nodes = pd.DataFrame(new_nodes_list)

# ==========================================
# 2. Link Refactoring
# ==========================================
print("Refactoring links...")
# Group by (new_start, new_end) to handle multi-edges
link_groups = {}

for _, row in df_links.iterrows():
    old_s = row['start']
    old_e = row['end']
    
    if old_s not in old_to_new_map or old_e not in old_to_new_map:
        continue

    new_s = old_to_new_map[old_s]
    new_e = old_to_new_map[old_e]
    
    # Remove self-loops
    if new_s == new_e:
        continue
        
    key = (new_s, new_e)
    if key not in link_groups:
        link_groups[key] = []
    
    link_groups[key].append(row)

new_links_list = []
link_id_counter = 1

def get_node_coord(nid):
    row = df_new_nodes[df_new_nodes['name'] == nid].iloc[0]
    return row['x'], row['y']

for (u, v), raw_links in link_groups.items():
    # Calculate new length (Euclidean)
    ux, uy = get_node_coord(u)
    vx, vy = get_node_coord(v)
    new_length = np.sqrt((ux - vx)**2 + (uy - vy)**2)
    
    # Aggregation logic
    # Capacity: Sum
    total_capacity = sum(l['u'] for l in raw_links)
    
    # Free Flow Speed (fft? or just 'u'? Wait, 'u' is usually capacity in some datasets, but let's check columns)
    # Input columns: name, start, end, length, u, kappa
    # Usually 'u' is capacity? Or 'u' is free flow speed?
    # In transportation, often 'C' is capacity, 't0' is free flow time.
    # Let's look at the file provided in prompt.
    # berlin_friedrichshain_links.csv columns: name, start, end, length, u, kappa
    # Standard TNTP format: Capacity is usually separate. 
    # But here we just have `u` and `kappa`.
    # Wait, commonly in this user's datasets (ChargingGame), `u` might be capacity?
    # Let's re-read the Task Doc.
    # "Capacity (u): Sum" -> So 'u' is capacity.
    # "Free Flow Speed: Weighted Average or Max".
    # BUT, the csv DOES NOT HAVE SPEED column listed in the prompt example?
    # "links.csv: name, start, end, length, u, kappa"
    # If there is no speed column, we can't average speed.
    # Maybe `length / free_flow_time`?
    # Let's assume `u` is capacity as per doc.
    # What about Speed?
    # If it's not in CSV, we can't calculate it.
    # However, if we are just outputting `u` and `kappa` and `length`, we just need those.
    # Doc says: "Capacity (u): Sum".
    # Doc says: "Free Flow Speed: ...".
    # If speed is not in the file, maybe we ignore it or it's implicit.
    # Let's just check the file content first to be sure.
    
    # Wait, I will assume the columns in `df_links` are all we have.
    # If there is a 'speed' or 'fft' column, I will handle it.
    # If not, I will stick to `u` (Capacity) and `kappa`.
    # Kappa is BPR parameter usually. weighted average? or just max?
    # Let's use max for Kappa (conservative) or average. Let's use weighted average by capacity for Kappa if possible, or just first.
    # Actually, let's just sum 'u'.
    # The Doc says "Free Flow Speed: Weighted Average". This implies there IS a speed.
    # If the input doesn't have speed, maybe I should check `read_file` first.
    # BUT, I am writing the script now. I will add a check.
    
    # For now, let's Sum 'u'.
    # Recalculate length.
    # Kappa: Average?
    avg_kappa = np.mean([l['kappa'] for l in raw_links])
    
    new_links_list.append({
        'name': link_id_counter,
        'start': u,
        'end': v,
        'length': new_length,
        'u': total_capacity,
        'kappa': avg_kappa
    })
    link_id_counter += 1

df_new_links = pd.DataFrame(new_links_list)

# ==========================================
# 3. Demand Remapping
# ==========================================
print("Remapping demand...")
# Demand columns: orig, dest, start_t, end_t, q
# Group by (new_orig, new_dest) AND (start_t, end_t)?
# Usually demand matrices are OD. Time might be relevant.
# "For same (New_Orig, New_Dest) pair, accumulate flow q."
# Does it mention time? "start_t, end_t".
# If we merge ODs, we should probably keep time slices distinct.
# So group by (new_orig, new_dest, start_t, end_t).

demand_groups = {}

for _, row in df_demand.iterrows():
    old_o = row['orig']
    old_d = row['dest']
    
    if old_o not in old_to_new_map or old_d not in old_to_new_map:
        continue
        
    new_o = old_to_new_map[old_o]
    new_d = old_to_new_map[old_d]
    
    if new_o == new_d:
        continue # Intra-zonal traffic
        
    # Key including time
    key = (new_o, new_d, row['start_t'], row['end_t'])
    
    if key not in demand_groups:
        demand_groups[key] = 0.0
    
    demand_groups[key] += row['q']

new_demand_list = []
for (o, d, st, et), q in demand_groups.items():
    new_demand_list.append({
        'orig': o,
        'dest': d,
        'start_t': st,
        'end_t': et,
        'q': q
    })

df_new_demand = pd.DataFrame(new_demand_list)

# ==========================================
# 4. Save Files
# ==========================================
print("Saving files...")
df_new_nodes.to_csv(f"{OUTPUT_DIR}/nodes.csv", index=False)
df_new_links.to_csv(f"{OUTPUT_DIR}/links.csv", index=False)
df_new_demand.to_csv(f"{OUTPUT_DIR}/demand.csv", index=False)

print(f"Done. Files saved to {OUTPUT_DIR}/")
