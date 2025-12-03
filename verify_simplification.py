import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
SIM_DIR = "berlin_friedrichshain_simplified"
ORIG_DIR = "berlin_friedrichshain"

sim_nodes_path = f"{SIM_DIR}/nodes.csv"
sim_links_path = f"{SIM_DIR}/links.csv"
sim_demand_path = f"{SIM_DIR}/demand.csv"
orig_demand_path = f"{ORIG_DIR}/{ORIG_DIR}_demand.csv"

# 1. Load Data
print("Loading data...")
nodes = pd.read_csv(sim_nodes_path)
links = pd.read_csv(sim_links_path)
sim_demand = pd.read_csv(sim_demand_path)
orig_demand = pd.read_csv(orig_demand_path)

# 2. Visualize Network
print("Generating network visualization...")
plt.figure(figsize=(12, 10))

# Plot Links
for _, link in links.iterrows():
    n_start = nodes[nodes['name'] == link['start']].iloc[0]
    n_end = nodes[nodes['name'] == link['end']].iloc[0]
    plt.plot([n_start.x, n_end.x], [n_start.y, n_end.y], 'gray', linewidth=0.5, alpha=0.7)

# Plot Nodes
plt.scatter(nodes.x, nodes.y, c='blue', s=10, label='Nodes', zorder=5)

# Highlight Charging Nodes (from settings if possible, but let's just plot base net first)
# If we want to show charging stations:
import json
try:
    with open(f"{SIM_DIR}/settings.json", 'r') as f:
        settings = json.load(f)
        charging_ids = [int(k) for k in settings.get("charging_nodes", {}).keys()]
        
    charging_nodes = nodes[nodes['name'].isin(charging_ids)]
    plt.scatter(charging_nodes.x, charging_nodes.y, c='red', s=50, marker='*', label='Charging Stations', zorder=10)
except:
    pass

plt.title("Simplified Berlin Friedrichshain Network")
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig("simplified_network_visualization.png")
print("Network visualization saved to 'simplified_network_visualization.png'")

# 3. OD Demand Analysis
print("\n=== OD Demand Analysis ===")

# Original Stats
orig_total_flow = orig_demand['q'].sum()
orig_od_pairs = len(orig_demand)
print(f"Original Total Flow: {orig_total_flow:.2f}")
print(f"Original OD Pairs:   {orig_od_pairs}")

# Simplified Stats
sim_total_flow = sim_demand['q'].sum()
sim_od_pairs = len(sim_demand)
print(f"Simplified Total Flow: {sim_total_flow:.2f}")
print(f"Simplified OD Pairs:   {sim_od_pairs}")

# Comparison
flow_loss = orig_total_flow - sim_total_flow
loss_percent = (flow_loss / orig_total_flow) * 100

print(f"\nFlow Reduction (Intra-zonal removed): {flow_loss:.2f} ({loss_percent:.2f}%)")
print(f"OD Pair Reduction: {orig_od_pairs - sim_od_pairs} pairs")

# Distribution check (Top 5 flows)
print("\nTop 5 OD Pairs (Simplified):")
print(sim_demand.sort_values('q', ascending=False).head(5).to_string(index=False))
