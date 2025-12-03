import tkinter as tk
from tkinter import ttk
import csv
import math
from collections import defaultdict

# --- Data Loading ---
def read_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_data():
    try:
        nodes = read_csv('berlin_friedrichshain/berlin_friedrichshain_nodes.csv')
        links = read_csv('berlin_friedrichshain/berlin_friedrichshain_links.csv')
        return nodes, links
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

# --- Core Logic ---
def simplify_network(nodes, links, threshold):
    # 1. Clustering
    parent = {n['name']: n['name'] for n in nodes}
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Coordinate lookup
    node_coords = {n['name']: (float(n['x']), float(n['y'])) for n in nodes}

    # Cluster
    node_names = list(node_coords.keys())
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            n1, n2 = node_names[i], node_names[j]
            x1, y1 = node_coords[n1]
            x2, y2 = node_coords[n2]
            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < threshold:
                union(n1, n2)

    # Group by cluster
    clusters = defaultdict(list)
    for n in nodes:
        root = find(n['name'])
        clusters[root].append(n)

    # 2. Create Super Nodes
    super_nodes = []
    old_to_new_map = {}
    
    for root, cluster_nodes in clusters.items():
        avg_x = sum(float(n['x']) for n in cluster_nodes) / len(cluster_nodes)
        avg_y = sum(float(n['y']) for n in cluster_nodes) / len(cluster_nodes)
        # Use root ID as the new super node ID for visualization stability
        super_nodes.append({'name': root, 'x': avg_x, 'y': avg_y, 'size': len(cluster_nodes)})
        for n in cluster_nodes:
            old_to_new_map[n['name']] = root

    # 3. Create Links
    super_links = []
    seen_links = set()
    
    for link in links:
        try:
            u_start = old_to_new_map[link['start']]
            u_end = old_to_new_map[link['end']]
        except KeyError:
            continue
            
        if u_start == u_end: continue # Skip self-loops
        
        # For visualization, we just need unique connections, not aggregated attributes
        # Use sorted tuple to avoid duplicates (A-B vs B-A) if undirected, 
        # but traffic is directed. Let's keep directed but unique.
        if (u_start, u_end) not in seen_links:
             super_links.append({'start': u_start, 'end': u_end})
             seen_links.add((u_start, u_end))

    return super_nodes, super_links

# --- GUI ---
class NetworkVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Network Simplification Visualizer")
        self.geometry("1000x800")

        self.nodes, self.links = load_data()
        self.scale_factor = 0.15 # Zoom level
        self.offset_x = 100
        self.offset_y = 100
        self.drag_start_x = 0
        self.drag_start_y = 0

        # Normalize coordinates for initial view
        if self.nodes:
            xs = [float(n['x']) for n in self.nodes]
            ys = [float(n['y']) for n in self.nodes]
            min_x, min_y = min(xs), min(ys)
            self.offset_x = -min_x * self.scale_factor + 50
            self.offset_y = -min_y * self.scale_factor + 50

        self._init_ui()
        self.update_visualization()

    def _init_ui(self):
        # Controls Panel
        panel = ttk.Frame(self)
        panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(panel, text="Merge Threshold (m):").pack(side=tk.LEFT)
        
        self.slider_var = tk.DoubleVar(value=50.0)
        self.slider = ttk.Scale(panel, from_=0, to=300, orient=tk.HORIZONTAL, variable=self.slider_var, length=400)
        self.slider.pack(side=tk.LEFT, padx=10)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Step buttons
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="-", width=2, command=lambda: self.change_threshold(-1)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="+", width=2, command=lambda: self.change_threshold(1)).pack(side=tk.LEFT)

        self.label_val = ttk.Label(panel, text="50.0 m")
        self.label_val.pack(side=tk.LEFT)
        
        self.label_stats = ttk.Label(panel, text="")
        self.label_stats.pack(side=tk.LEFT, padx=20)

        # Canvas
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<MouseWheel>", self.on_zoom) # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)   # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)   # Linux scroll down

    def on_slider_release(self, event):
        val = self.slider_var.get()
        self.label_val.config(text=f"{val:.1f} m")
        self.update_visualization()

    def change_threshold(self, delta):
        val = self.slider_var.get() + delta
        val = max(0, min(300, val)) # Clamp
        self.slider_var.set(val)
        self.label_val.config(text=f"{val:.1f} m")
        self.update_visualization()

    def world_to_screen(self, x, y):
        # Invert Y because screen coords are y-down
        h = self.canvas.winfo_height()
        if h < 10: h = 800 # Fallback during init
        
        sx = x * self.scale_factor + self.offset_x
        sy = h - (y * self.scale_factor + self.offset_y) 
        return sx, sy

    def update_visualization(self):
        threshold = self.slider_var.get()
        s_nodes, s_links = simplify_network(self.nodes, self.links, threshold)
        
        self.label_stats.config(text=f"Nodes: {len(self.nodes)} -> {len(s_nodes)} | Links: {len(self.links)} -> {len(s_links)}")
        
        self.canvas.delete("all")
        
        node_pos = {n['name']: (n['x'], n['y']) for n in s_nodes}

        # Draw Links
        for link in s_links:
            try:
                x1, y1 = node_pos[link['start']]
                x2, y2 = node_pos[link['end']]
                sx1, sy1 = self.world_to_screen(x1, y1)
                sx2, sy2 = self.world_to_screen(x2, y2)
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#ccc", width=1)
            except KeyError:
                pass

        # Draw Nodes
        for n in s_nodes:
            sx, sy = self.world_to_screen(n['x'], n['y'])
            
            # Radius depends on whether it's a merged node (size > 1)
            r = 3 if n['size'] == 1 else 6
            color = "#888" if n['size'] == 1 else "red" # Red for centroids (merged nodes)
            
            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=color, outline="")
            
            if n['size'] > 1:
                 self.canvas.create_text(sx, sy-10, text=str(n['size']), font=("Arial", 8), fill="blue")

    # --- Interaction ---
    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag_motion(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        # Y is inverted in drawing, so dragging down (positive dy) should decrease offset_y if we want map to move down
        # Wait, if I drag down, I want the map to move down. 
        # sy = H - (y*s + off_y)
        # sy_new = sy + dy
        # H - (y*s + off_y_new) = H - (y*s + off_y) + dy
        # -off_y_new = -off_y + dy => off_y_new = off_y - dy
        self.offset_x += dx
        self.offset_y -= dy 
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_visualization()

    def on_zoom(self, event):
        # Windows: event.delta, Linux: event.num
        scale_mult = 1.1
        if event.num == 5 or event.delta < 0:
            self.scale_factor /= scale_mult
        else:
            self.scale_factor *= scale_mult
        self.update_visualization()

if __name__ == "__main__":
    app = NetworkVisualizer()
    app.mainloop()
