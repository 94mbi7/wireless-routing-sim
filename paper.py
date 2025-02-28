import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gym

# Load dataset
data = pd.DataFrame({
    'Device ID': ['D1', 'D1', 'D2', 'D3', 'D4', 'D5'],
    'Battery Level (%)': [80, 80, 65, 90, 75, 50],
    'Residual Battery Life (hrs)': [10, 10, 7, 12, 9, 4],
    'Mobility (m/s)': [0, 0, 1.2, 0, 1.5, 0],
    'Bandwidth (Mbps)': [5, 5, 4, 6, 3, 7],
    'Neighbor ID': ['D2', 'D3', 'D4', 'D5', 'D5', 'D1'],
    'Transmission Time (ms)': [5, 8, 6, 10, 7, 9],
    'Packet Loss Rate (%)': [1.2, 2.5, 0.8, 3.1, 1.5, 2.0],
    'RSSI (dBm)': [-55, -60, -50, -65, -58, -62],
    'Interference (dB)': [3, 5, 2, 6, 4, 5],
    'Hop Count': [2, 3, 1, 4, 2, 3],
    'Congestion Level (ms)': [10, 15, 8, 20, 12, 18],
    'Weather Condition': ['Clear', 'Rain', 'Foggy', 'Clear', 'Windy', 'Rain']
})

def create_network_from_data(df):
    """Create network graph from device dataset"""
    G = nx.Graph()
    
    # Add nodes with device attributes
    for device_id in pd.unique(df['Device ID']):
        device_data = df[df['Device ID'] == device_id].iloc[0]
        G.add_node(device_data['Device ID'],
                   battery=device_data['Battery Level (%)'],
                   residual_battery=device_data['Residual Battery Life (hrs)'],
                   mobility=device_data['Mobility (m/s)'],
                   bandwidth=device_data['Bandwidth (Mbps)'],
                   pos=(np.random.uniform(0, 10), np.random.uniform(0, 10)))

    # Add edges with connection metrics
    for _, row in df.iterrows():
        G.add_edge(row['Device ID'], row['Neighbor ID'],
                   transmission_time=row['Transmission Time (ms)']/1000,
                   packet_loss=row['Packet Loss Rate (%)']/100,
                   rssi=row['RSSI (dBm)'],
                   interference=row['Interference (dB)'],
                   hop_count=row['Hop Count'],
                   congestion=row['Congestion Level (ms)']/1000,
                   weather=row['Weather Condition'],
                   weight=calculate_edge_weight(row))
    return G

def calculate_edge_weight(row):
    """Composite weight calculation using all parameters"""
    base_weight = row['Transmission Time (ms)'] + row['Congestion Level (ms)']
    
    weather_penalty = {
        'Clear': 0,
        'Rain': row['Packet Loss Rate (%)'] * 2,
        'Foggy': row['Interference (dB)'] * 0.8,
        'Windy': row['RSSI (dBm)'] * -0.2
    }[row['Weather Condition']]
    
    reliability_penalty = (
        (100 - row['Battery Level (%)']) * 0.1 +
        row['Hop Count'] * 5 +
        row['Interference (dB)'] * 0.5
    )
    
    return (base_weight + weather_penalty + reliability_penalty) / 1000

def update_network(G):
    """Enhanced dynamic updates based on real-world factors"""
    for u, v in G.edges:
        # Update congestion based on weather
        weather = G.edges[u, v]['weather']
        congestion_change = {
            'Clear': np.random.uniform(-2, 1),
            'Rain': np.random.uniform(0, 5),
            'Foggy': np.random.uniform(1, 3),
            'Windy': np.random.uniform(-1, 2)
        }[weather]
        
        G.edges[u, v]['congestion'] = np.clip(
            G.edges[u, v]['congestion'] + congestion_change,
            0.005, 0.050  # Keep in seconds
        )
        
        # Add RSSI to the parameter dictionary
        G.edges[u, v]['weight'] = calculate_edge_weight({
            'Transmission Time (ms)': G.edges[u, v]['transmission_time']*1000,
            'Congestion Level (ms)': G.edges[u, v]['congestion']*1000,
            'Weather Condition': weather,
            'Packet Loss Rate (%)': G.edges[u, v]['packet_loss']*100,
            'Interference (dB)': G.edges[u, v]['interference'],
            'RSSI (dBm)': G.edges[u, v]['rssi'],  # Add this line
            'Battery Level (%)': (G.nodes[u]['battery'] + G.nodes[v]['battery'])/2,
            'Hop Count': G.edges[u, v]['hop_count']
        })

    for node in G.nodes:
        G.nodes[node]['battery'] = max(0, G.nodes[node]['battery'] - 
            (0.1 * G.nodes[node]['mobility'] + 0.05 * G.nodes[node]['bandwidth']))
    
    return G

class EnhancedWirelessRoutingEnv(gym.Env):
    def __init__(self, G, source, target):
        super().__init__()
        self.G = G
        self.source = source
        self.target = target
        self.observation_space = gym.spaces.Dict({
            'current_node': gym.spaces.Discrete(len(G.nodes)),
            'neighbors': gym.spaces.Dict({
                'battery': gym.spaces.Box(low=0, high=100, shape=(1,)),
                'congestion': gym.spaces.Box(low=5, high=50, shape=(1,)),
                'packet_loss': gym.spaces.Box(low=0, high=1, shape=(1,))
            })
        })
        self.reset()

    def reset(self):
        self.state = {
            'current_node': self.source,
            'neighbors': self._get_neighbor_info(self.source)
        }
        return self.state

    def step(self, action):
        current_node = self.state['current_node']
        neighbors = list(self.G.neighbors(current_node))
        
        if not neighbors:
            return self.state, -10, True, {}
            
        chosen_neighbor = neighbors[action]
        edge_data = self.G[current_node][chosen_neighbor]
        battery_cost = (self.G.nodes[current_node]['battery'] * 0.01 +
                       self.G.nodes[chosen_neighbor]['battery'] * 0.005)
        reward = -(edge_data['weight'] + battery_cost + edge_data['packet_loss'])
        
        self.state = {
            'current_node': chosen_neighbor,
            'neighbors': self._get_neighbor_info(chosen_neighbor)
        }
        
        done = chosen_neighbor == self.target
        return self.state, reward, done, {}

    def _get_neighbor_info(self, node):
        return {
            neighbor: {
                'battery': self.G.nodes[neighbor]['battery'],
                'congestion': self.G.edges[node, neighbor]['congestion']*1000,
                'packet_loss': self.G.edges[node, neighbor]['packet_loss']
            }
            for neighbor in self.G.neighbors(node)
        }

def adaptive_aco(G, source, target, pheromones, ants=5, decay=0.1):
    best_path, best_cost = None, float('inf')
    for _ in range(ants):
        try:
            path = nx.shortest_path(G, source, target, 
                                  weight=lambda u,v,d: d['weight']/pheromones[(u,v)])
            cost = sum(G[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
            if cost < best_cost:
                best_path, best_cost = path, cost
            for u,v in zip(path[:-1], path[1:]):
                pheromones[(u,v)] = pheromones[(u,v)]*(1-decay) + 1/cost
        except nx.NetworkXNoPath:
            continue
    return best_path

def q_learning_path(Q, G, source, target):
    path = [source]
    current = source
    while current != target and len(path) < len(G.nodes):
        neighbors = list(G[current])
        if not neighbors: break
        action = np.argmax(Q[current][:len(neighbors)])
        current = neighbors[action]
        path.append(current)
    return path if current == target else None

def draw_network(G, pos, paths):
    plt.clf()
    ax = plt.gca()
    
    # Node styling
    node_colors = [G.nodes[n]['battery'] for n in G.nodes]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  cmap=plt.cm.YlGn, node_size=800, 
                                  vmin=0, vmax=100)
    
    # Edge styling (only if edges exist)
    if G.edges:
        edge_colors = [G.edges[e]['congestion']*1000 for e in G.edges]
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                      edge_cmap=plt.cm.Reds, width=2, 
                                      edge_vmin=5, edge_vmax=50)
        
        # Add colorbar only if edges exist
        plt.colorbar(edges, label='Congestion (ms)', ax=ax)
    
    # Labels
    node_labels = {n: f"{n}\n{G.nodes[n]['battery']}%\n{G.nodes[n]['residual_battery']}h"
                  for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Path highlighting
    colors = {'Dijkstra': 'blue', 'ACO': 'red', 'Q-Learning': 'green'}
    for algo, path in paths.items():
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                  edge_color=colors[algo], width=4)
    
    plt.title("Wireless Network Routing Simulation")
    plt.axis('off')
    plt.pause(0.1)

def dynamic_path_finding(G, pos, source, target, steps=10):
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    
    env = EnhancedWirelessRoutingEnv(G, source, target)
    Q = defaultdict(lambda: np.zeros(len(G.nodes)))
    pheromones = defaultdict(lambda: 1.0)
    
    for step in range(steps):
        G = update_network(G)
        print(f"\nStep {step+1}/{steps}")
        
        # Calculate paths
        paths = {}
        try:
            paths['Dijkstra'] = nx.shortest_path(G, source, target, weight='weight')
            print(f"Dijkstra path: {paths['Dijkstra']}")
        except nx.NetworkXNoPath:
            paths['Dijkstra'] = None
            
        paths['ACO'] = adaptive_aco(G, source, target, pheromones)
        print(f"ACO path: {paths['ACO']}")
        
        paths['Q-Learning'] = q_learning_path(Q, G, source, target)
        print(f"Q-Learning path: {paths['Q-Learning']}")
        
        # Q-learning training
        state = env.reset()
        done = False
        while not done:
            neighbors = list(env.G[state['current_node']])
            if not neighbors: break
            action = (np.random.randint(len(neighbors)) if np.random.rand() < 0.1 
                     else np.argmax(Q[state['current_node']][:len(neighbors)]))
            next_state, reward, done, _ = env.step(action)
            Q[state['current_node']][action] += 0.1 * (
                reward + 0.9 * np.max(Q[next_state['current_node']]) - 
                Q[state['current_node']][action]
            )
            state = next_state
        
        draw_network(G, pos, paths)
        plt.pause(1)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    G = create_network_from_data(data)
    
    # Set fixed positions for better visualization
    fixed_pos = {
        'D1': (2, 8), 'D2': (5, 6), 'D3': (8, 8),
        'D4': (4, 4), 'D5': (7, 2)
    }
    nx.set_node_attributes(G, fixed_pos, 'pos')
    
    dynamic_path_finding(G, fixed_pos, 'D1', 'D5', steps=10)