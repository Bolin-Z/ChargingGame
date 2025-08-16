from src.utils import load_network, compute_routes

W = load_network("./siouxfalls", "siouxfalls")
dict_od_to_routes = compute_routes(W, 10)
print("charging_routes:")
for od, routes in dict_od_to_routes["charging"].items():
    o, d = od
    print(f"\t{o} -> {d}: {len(routes)}")
print("uncharging_routes:")
for od, routes in dict_od_to_routes["uncharging"].items():
    o, d = od
    print(f"\t{o} -> {d}: {len(routes)}")