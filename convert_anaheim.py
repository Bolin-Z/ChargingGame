"""
Anaheim TNTP -> Project Standard Format Converter

Converts Anaheim transportation network data from TNTP format to project standard CSV format.
Source: construct_dataset/Anaheim/
Target: data/anaheim/

Unit conversions:
- Distance: feet -> meters (x0.3048)
- Speed: ft/min -> m/s (x0.00508 = 0.3048/60)
"""

import json
import csv
import re
from pathlib import Path


# Conversion constants
FEET_TO_METERS = 0.3048
FT_PER_MIN_TO_M_PER_S = 0.3048 / 60  # = 0.00508


def parse_geojson_nodes(geojson_path: str) -> dict:
    """Parse GeoJSON file to extract node coordinates.

    Returns:
        dict: {node_id: (x, y)} where x=longitude, y=latitude
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = {}
    for feature in data['features']:
        node_id = feature['properties']['id']
        coords = feature['geometry']['coordinates']
        x, y = coords[0], coords[1]  # longitude, latitude
        nodes[node_id] = (x, y)

    return nodes


def parse_tntp_network(net_path: str) -> list:
    """Parse TNTP network file to extract link data.

    Returns:
        list of dicts with keys: init_node, term_node, capacity, length, speed
    """
    links = []

    with open(net_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find data section (after END OF METADATA)
    lines = content.split('\n')
    data_started = False

    for line in lines:
        line = line.strip()
        if '<END OF METADATA>' in line:
            data_started = True
            continue

        if not data_started:
            continue

        # Skip empty lines and comments
        if not line or line.startswith('~'):
            continue

        # Parse link data
        # Format: init_node term_node capacity length free_flow_time b power speed toll link_type ;
        parts = line.replace('\t', ' ').split()
        if len(parts) >= 9 and parts[-1] == ';':
            try:
                link = {
                    'init_node': int(parts[0]),
                    'term_node': int(parts[1]),
                    'capacity': float(parts[2]),
                    'length': float(parts[3]),  # in feet
                    'free_flow_time': float(parts[4]),  # in minutes
                    'speed': float(parts[7]),  # in ft/min
                }
                links.append(link)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")

    return links


def parse_tntp_trips(trips_path: str) -> list:
    """Parse TNTP trips file to extract OD demand.

    Returns:
        list of tuples: (origin, destination, demand)
    """
    demands = []

    with open(trips_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    current_origin = None
    data_started = False

    for line in lines:
        line = line.strip()

        if '<END OF METADATA>' in line:
            data_started = True
            continue

        if not data_started:
            continue

        # Check for new origin
        if line.startswith('Origin'):
            parts = line.split()
            current_origin = int(parts[1])
            continue

        if current_origin is None or not line:
            continue

        # Parse destination:demand pairs
        # Format: dest : demand ; dest : demand ; ...
        pairs = line.split(';')
        for pair in pairs:
            pair = pair.strip()
            if not pair or ':' not in pair:
                continue

            try:
                dest_str, demand_str = pair.split(':')
                dest = int(dest_str.strip())
                demand = float(demand_str.strip())
                if demand > 0:
                    demands.append((current_origin, dest, demand))
            except ValueError:
                continue

    return demands


def calculate_merge_priority(speed_mps: float) -> int:
    """Calculate merge priority based on speed (similar to siouxfalls).

    Higher speed -> higher priority
    """
    if speed_mps >= 20:
        return 6
    elif speed_mps >= 15:
        return 5
    elif speed_mps >= 10:
        return 4
    elif speed_mps >= 6:
        return 3
    elif speed_mps >= 4:
        return 2
    else:
        return 1


def convert_anaheim():
    """Main conversion function."""
    # Paths
    base_dir = Path(__file__).parent
    source_dir = base_dir / 'Anaheim'
    target_dir = base_dir.parent / 'data' / 'anaheim'

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Anaheim Dataset Conversion")
    print("=" * 60)

    # 1. Parse source data
    print("\n[1/4] Parsing source data...")

    nodes = parse_geojson_nodes(source_dir / 'anaheim_nodes.geojson')
    print(f"  - Nodes: {len(nodes)}")

    links = parse_tntp_network(source_dir / 'Anaheim_net.tntp')
    print(f"  - Links: {len(links)}")

    demands = parse_tntp_trips(source_dir / 'Anaheim_trips.tntp')
    print(f"  - OD pairs: {len(demands)}")
    total_demand = sum(d[2] for d in demands)
    print(f"  - Total demand: {total_demand:.2f}")

    # 2. Convert and write nodes
    print("\n[2/4] Converting nodes...")
    nodes_file = target_dir / 'anaheim_nodes.csv'
    with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'x', 'y'])
        for node_id in sorted(nodes.keys()):
            x, y = nodes[node_id]
            writer.writerow([node_id, x, y])
    print(f"  - Written: {nodes_file}")

    # 3. Convert and write links
    print("\n[3/4] Converting links...")
    links_file = target_dir / 'anaheim_links.csv'
    with open(links_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'start', 'end', 'length', 'u', 'kappa', 'merge_priority'])

        for link in links:
            name = f"{link['init_node']}-{link['term_node']}"
            start = link['init_node']
            end = link['term_node']
            length = link['length'] * FEET_TO_METERS  # feet -> meters
            u = link['speed'] * FT_PER_MIN_TO_M_PER_S  # ft/min -> m/s
            kappa = 0.2  # standard jam density
            merge_priority = calculate_merge_priority(u)

            writer.writerow([name, start, end, round(length, 2), round(u, 4), kappa, merge_priority])

    print(f"  - Written: {links_file}")

    # 4. Convert and write demand
    print("\n[4/4] Converting demand...")
    demand_file = target_dir / 'anaheim_demand.csv'

    # Scale demand to vehicles per second (similar to siouxfalls)
    # Anaheim total: ~104,694 trips, simulation_time: 9600s
    # We need to convert to a rate (vehicles per second)
    simulation_time = 9600  # seconds

    with open(demand_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['orig', 'dest', 'start_t', 'end_t', 'q'])

        for orig, dest, demand in demands:
            # Convert total trips to rate (trips per second during demand period)
            # Demand is released during first half of simulation
            start_t = 0
            end_t = simulation_time // 2  # 4800 seconds
            q = demand / end_t  # vehicles per second

            writer.writerow([orig, dest, start_t, end_t, round(q, 6)])

    print(f"  - Written: {demand_file}")

    # 5. Create settings file
    print("\n[5/5] Creating settings file...")
    settings = {
        "network_name": "anaheim",
        "simulation_time": 9600,
        "deltan": 5,
        "demand_multiplier": 1.0,

        "charging_car_rate": 0.1,
        "charging_link_length": 3000,
        "charging_link_free_flow_speed": 10,

        "charging_periods": 8,
        "charging_nodes": {},  # To be determined after flow analysis

        "routes_per_od": 10,
        "time_value_coefficient": 0.005,
        "charging_demand_per_vehicle": 50,
        "ue_convergence_threshold": 1.0,
        "ue_max_iterations": 100,
        "ue_swap_probability": 0.05
    }

    settings_file = target_dir / 'anaheim_settings.json'
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    print(f"  - Written: {settings_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {target_dir}")
    print(f"\nFiles created:")
    print(f"  - anaheim_nodes.csv ({len(nodes)} nodes)")
    print(f"  - anaheim_links.csv ({len(links)} links)")
    print(f"  - anaheim_demand.csv ({len(demands)} OD pairs)")
    print(f"  - anaheim_settings.json")

    # Statistics
    print(f"\nNetwork Statistics:")
    print(f"  - Total nodes: {len(nodes)}")
    print(f"  - Total links: {len(links)}")
    print(f"  - Zones (OD zones): 38")
    print(f"  - Total trips: {total_demand:.2f}")

    # Link length statistics (in meters)
    lengths = [link['length'] * FEET_TO_METERS for link in links]
    print(f"\nLink Lengths (meters):")
    print(f"  - Min: {min(lengths):.2f}")
    print(f"  - Max: {max(lengths):.2f}")
    print(f"  - Avg: {sum(lengths)/len(lengths):.2f}")

    # Speed statistics (in m/s)
    speeds = [link['speed'] * FT_PER_MIN_TO_M_PER_S for link in links]
    print(f"\nLink Speeds (m/s):")
    print(f"  - Min: {min(speeds):.2f}")
    print(f"  - Max: {max(speeds):.2f}")
    print(f"  - Avg: {sum(speeds)/len(speeds):.2f}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Run test_anaheim.py to verify UXSim loading")
    print("  2. Adjust demand_multiplier if needed")
    print("  3. Run flow analysis for charging station placement")
    print("=" * 60)


if __name__ == '__main__':
    convert_anaheim()
