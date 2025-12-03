"""
Test script to verify berlin_friedrichshain_simplified dataset loads and runs in UXSim.
Includes animation generation for visualization.
"""
import csv
import json
from uxsim import World

def test_uxsim_load(demand_multiplier=1.0):
    """Test loading the simplified Berlin network into UXSim with animation output.

    Args:
        demand_multiplier: Factor to scale demand (e.g., 50.0 for 50x demand)
    """

    network_dir = "berlin_friedrichshain_simplified"

    # Load settings
    with open(f"{network_dir}/settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    print(f"=== Testing {settings['network_name']} ===")
    print(f"Simulation time: {settings['simulation_time']}s")
    print(f"Delta N: {settings['deltan']}")
    print(f"Charging stations: {len(settings['charging_nodes'])}")
    print(f"Demand multiplier: {demand_multiplier}x")

    # Create UXSim World
    # save_mode=1: save visualization results
    # show_mode=0: don't display plots during simulation (set to 1 for Jupyter)
    W = World(
        name="berlin_simplified_test",
        deltan=settings["deltan"],
        tmax=settings["simulation_time"],
        print_mode=1,
        save_mode=1,  # Enable saving visualization results
        show_mode=0,  # Set to 1 to display plots
        random_seed=42
    )

    # Load nodes
    nodes = {}
    with open(f"{network_dir}/nodes.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_name = row["name"]
            x = float(row["x"])
            y = float(row["y"])
            W.addNode(node_name, x, y)
            nodes[node_name] = (x, y)

    print(f"Loaded {len(nodes)} nodes")

    # Load links
    link_count = 0
    with open(f"{network_dir}/links.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_name = row["name"]
            start = row["start"]
            end = row["end"]
            length = float(row["length"])
            capacity = float(row["u"])  # 'u' is free flow speed in the format
            kappa = float(row["kappa"])  # jam density

            W.addLink(
                name=link_name,
                start_node=start,
                end_node=end,
                length=length,
                free_flow_speed=capacity,  # 'u' column is free flow speed
                jam_density=kappa,
                number_of_lanes=1
            )
            link_count += 1

    print(f"Loaded {link_count} links")

    # Load demands
    demand_count = 0
    with open(f"{network_dir}/demand.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row["orig"]
            dest = row["dest"]
            start_t = float(row["start_t"])
            end_t = float(row["end_t"])
            q = float(row["q"])  # vehicles per second

            if q > 0:
                W.adddemand(orig, dest, start_t, end_t, q * demand_multiplier)
                demand_count += 1

    print(f"Loaded {demand_count} OD demands (scaled by {demand_multiplier}x)")

    # Run simulation
    print("\n=== Running simulation ===")
    W.exec_simulation()

    # Print results
    print("\n=== Simulation Results ===")
    W.analyzer.print_simple_stats()

    # Generate visualizations
    print("\n=== Generating visualizations ===")

    # 1. Static network visualization
    print("Generating network plot...")
    W.analyzer.network(figsize=(12, 12), network_font_size=0)

    # 2. Network animation (traffic flow over time)
    # animation_speed_inverse: higher = slower animation
    # timestep_skip: frames to skip (larger = faster but coarser)
    # detailed=0: link-level view, detailed=1: cell-level detail
    print("Generating network animation (this may take a while)...")
    W.analyzer.network_anim(
        animation_speed_inverse=15,
        timestep_skip=8,
        detailed=0,
        figsize=(12, 12),
        network_font_size=0,
        file_name="berlin_network_anim.gif"
    )

    # 3. Time-space diagram for a sample link (if needed)
    # W.analyzer.time_space_diagram_traj_links([link_names])

    # 4. MFD (Macroscopic Fundamental Diagram)
    print("Generating MFD...")
    W.analyzer.macroscopic_fundamental_diagram()

    # Verification
    print("\n=== Verification ===")
    print(f"Total vehicles generated: {len(W.VEHICLES) * W.DELTAN}")

    completed = sum(1 for v in W.VEHICLES.values() if v.state == "end")
    print(f"Completed trips (platoons): {completed}")
    print(f"Completed trips (vehicles): {completed * W.DELTAN}")

    if completed > 0:
        print(f"\n[SUCCESS] Network loaded and simulation completed!")
        print(f"Output saved to: out{W.name}/")
        return True
    else:
        print("\n[WARNING] No vehicles completed their trips. Check network connectivity.")
        return False


def test_fancy_animation():
    """Generate fancy vehicle trajectory animation (optional, more computationally intensive)."""

    network_dir = "berlin_friedrichshain_simplified"

    with open(f"{network_dir}/settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    # Use shorter simulation for fancy animation demo
    W = World(
        name="berlin_fancy_test",
        deltan=settings["deltan"],
        tmax=1800,  # 30 minutes for demo
        print_mode=1,
        save_mode=1,
        show_mode=0,
        random_seed=42
    )

    # Load network (same as above)
    with open(f"{network_dir}/nodes.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            W.addNode(row["name"], float(row["x"]), float(row["y"]))

    with open(f"{network_dir}/links.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            W.addLink(
                name=row["name"],
                start_node=row["start"],
                end_node=row["end"],
                length=float(row["length"]),
                free_flow_speed=float(row["u"]),
                jam_density=float(row["kappa"]),
                number_of_lanes=1
            )

    # Load demands (first 30 min only)
    with open(f"{network_dir}/demand.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = float(row["q"])
            if q > 0:
                W.adddemand(row["orig"], row["dest"], 0, 1800, q)

    W.exec_simulation()
    W.analyzer.print_simple_stats()

    # Fancy animation with vehicle trajectories
    print("\nGenerating fancy animation with vehicle trajectories...")
    W.analyzer.network_fancy(
        animation_speed_inverse=15,
        sample_ratio=0.3,  # Show 30% of vehicles
        interval=5,
        trace_length=3,
        speed_coef=2,
        figsize=(12, 12),
        file_name="berlin_fancy_anim.gif"
    )

    print(f"Fancy animation saved to: out{W.name}/")


if __name__ == "__main__":
    # Run basic test with demand multiplier
    # 1x = no congestion (100% completion, 28 m/s)
    # 3x = moderate (100% completion, 10.5 m/s, delay 78%)
    # 4x = heavy (81% completion, 4.1 m/s, delay 93%)
    # 5x = severe (42% completion, 2.5 m/s)
    success = test_uxsim_load(demand_multiplier=3.5)

    if success:
        # Optionally run fancy animation (uncomment if desired)
        # print("\n" + "="*50)
        # print("Running fancy animation test...")
        # test_fancy_animation()
        pass
