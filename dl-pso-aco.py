import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    def __init__(self, num_nodes, area_size=80, comm_range=40):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.comm_range = comm_range
        self.base_station = 0
        self.node_positions = {self.base_station: (area_size/2, area_size/2)}
        self.node_energy = {self.base_station: float('inf')}
        self.traffic_load = {self.base_station: 0}
        self.topology_map = nx.Graph()
        self._initialize_nodes()

    def _initialize_nodes(self):
        for i in range(1, self.num_nodes):
            x = np.random.uniform(self.area_size/2 - self.comm_range*0.7,
                                 self.area_size/2 + self.comm_range*0.7)
            y = np.random.uniform(self.area_size/2 - self.comm_range*0.7,
                                 self.area_size/2 + self.comm_range*0.7)
            self.node_positions[i] = (x, y)
            self.node_energy[i] = 1.0
            self.traffic_load[i] = 0
        self._update_topology(initialize=True)

    def _distance(self, u, v):
        return np.hypot(
            self.node_positions[u][0] - self.node_positions[v][0],
            self.node_positions[u][1] - self.node_positions[v][1]
        )

    def _simulate_node_failures(self, training_mode=False):
        if training_mode:
            return
        
        active_nodes = list(self.topology_map.nodes())
        for node in active_nodes:
            if node == self.base_station:
                continue
            fail_prob = 0.05 * (1 - self.node_energy[node])
            if np.random.rand() < fail_prob:
                self.topology_map.remove_node(node)
                del self.node_positions[node]
                del self.node_energy[node]
                del self.traffic_load[node]
                
    def _update_topology(self, initialize=False, training_mode=False):
        if not initialize:
            self.topology_map.clear_edges()
        
        valid_nodes = list(self.node_positions.keys())
        new_edges = []
        
        for node in valid_nodes:
            self.topology_map.add_node(node, 
                                      energy=self.node_energy[node],
                                      position=self.node_positions[node])
        
        for u in valid_nodes:
            for v in valid_nodes:
                if u != v and self._distance(u, v) <= self.comm_range:
                    new_edges.append((u, v, {'weight': self._distance(u, v)}))
        
        self.topology_map.add_edges_from(new_edges)
        self._simulate_node_failures(training_mode=training_mode)

    def update_state(self, training_mode=False):
        for node in list(self.node_energy.keys()):
            if node != self.base_station:
                energy_loss = 0.002 + 0.001 * self.traffic_load[node]
                self.node_energy[node] = max(0, self.node_energy[node] - energy_loss)
        
        self.traffic_load = {n: np.random.poisson(2) for n in self.node_energy}
        self._update_topology(training_mode=training_mode)

class DLModel(tf.keras.Model):
    def __init__(self, num_features, num_nodes):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu')
        ])
        self.decoder = tf.keras.layers.Dense(num_nodes, activation='softmax')
        
    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

class PSOOptimizer:
    def __init__(self, num_particles=30, inertia=0.8, cognitive=1.7, social=1.7):
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        
    class Particle:
        def __init__(self, dim):
            self.position = np.random.uniform(0.1, 0.9, dim)
            self.velocity = np.zeros(dim)
            self.best_position = self.position.copy()
            self.best_score = float('inf')
    
    def optimize(self, cost_func, max_iter=100):
        particles = [self.Particle(3) for _ in range(self.num_particles)]
        global_best = (None, float('inf'))
        
        for _ in range(max_iter):
            for p in particles:
                current_score = cost_func(p.position)
                if current_score < p.best_score:
                    p.best_position = p.position.copy()
                    p.best_score = current_score
                if current_score < global_best[1]:
                    global_best = (p.position.copy(), current_score)
            
            for p in particles:
                inertia_term = self.inertia * p.velocity
                cognitive_term = self.cognitive * np.random.rand() * (p.best_position - p.position)
                social_term = self.social * np.random.rand() * (global_best[0] - p.position)
                p.velocity = inertia_term + cognitive_term + social_term
                p.position = np.clip(p.position + p.velocity, 0.1, 0.9)
                
        return global_best[0]

class ACOSolver:
    def __init__(self, num_nodes, evaporation_rate=0.2):
        self.num_nodes = num_nodes
        self.evaporation_rate = evaporation_rate
        self.pheromones = np.ones((num_nodes, num_nodes)) * 0.1
        self.pso_weights = np.array([0.33, 0.33, 0.34])
        self.epsilon = 1e-10
        
    def apply_pso_parameters(self, pso_params):
        total = np.sum(pso_params) + self.epsilon
        self.pso_weights = np.array(pso_params) / total
        
    def update_pheromones(self, paths, energy_cost_func):
        self.pheromones *= (1 - self.evaporation_rate)
        for path in paths:
            if not path or path[-1] != 0 or len(path) < 2:
                continue
            
            cost = energy_cost_func(path)
            if np.isfinite(cost):
                delta = 1 / (1 + cost)
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    self.pheromones[u][v] += delta
                    self.pheromones[v][u] += delta
                    
    def find_paths(self, network_graph, num_ants=15):
        paths = []
        valid_nodes = list(network_graph.nodes())
        node_energy = nx.get_node_attributes(network_graph, 'energy')
        
        for _ in range(num_ants):
            current = np.random.choice(valid_nodes)
            path = [current]
            visited = set()
            
            while current != 0 and len(path) < self.num_nodes:
                visited.add(current)
                neighbors = [n for n in network_graph.neighbors(current) 
                           if n not in visited]
                
                if not neighbors:
                    break
                
                distances = []
                for n in neighbors:
                    try:
                        dist = network_graph[current][n]['weight']
                        distances.append(max(dist, self.epsilon))
                    except KeyError:
                        distances.append(self.epsilon)
                
                pheromones = self.pheromones[current, neighbors]
                energies = [node_energy.get(n, 0.5) for n in neighbors]
                
                heuristic = (
                    self.pso_weights[0] * pheromones +
                    self.pso_weights[1] * (1/np.array(distances)) +
                    self.pso_weights[2] * np.array(energies)
                )
                
                heuristic = np.nan_to_num(heuristic, nan=self.epsilon, posinf=1e5, neginf=0)
                heuristic_sum = np.sum(heuristic) + self.epsilon
                
                if heuristic_sum <= 0:
                    probs = np.ones_like(heuristic)/len(heuristic)
                else:
                    probs = heuristic / heuristic_sum
                
                current = np.random.choice(neighbors, p=probs)
                path.append(current)
            
            paths.append(path)
        return paths

class HybridRouter:
    def __init__(self, num_nodes):
        self.monitor = NetworkMonitor(num_nodes)
        self.dl_model = DLModel(num_features=3, num_nodes=num_nodes)
        self.pso = PSOOptimizer()
        self.aco = ACOSolver(num_nodes)
        
    def generate_training_data(self, num_samples=2000):
        X, y = [], []
        attempts = 0
        max_attempts = num_samples * 3
        
        while len(X) < num_samples and attempts < max_attempts:
            self.monitor.update_state(training_mode=True)
            active_nodes = [n for n in self.monitor.node_energy 
                           if n != self.monitor.base_station]
            attempts += 1
            
            if not active_nodes:
                continue
                
            for node in active_nodes:
                try:
                    distance = nx.shortest_path_length(
                        self.monitor.topology_map,
                        node,
                        self.monitor.base_station,
                        weight='weight'
                    )
                except nx.NetworkXNoPath:
                    distance = self.monitor.area_size * np.sqrt(2)
                
                features = [
                    self.monitor.node_energy[node],
                    self.monitor.traffic_load[node],
                    distance / (self.monitor.area_size * np.sqrt(2))
                ]
                
                try:
                    path = nx.shortest_path(
                        self.monitor.topology_map,
                        node,
                        self.monitor.base_station,
                        weight=lambda u,v,d: 0.4*d['weight'] + 0.2*self.monitor.traffic_load[v]
                    )
                    target = path[1] if len(path) > 1 else 0
                except nx.NetworkXNoPath:
                    target = 0
                
                X.append(features)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_dl_model(self, epochs=50, batch_size=64):
        X_train, y_train = self.generate_training_data()
        if len(X_train) < 100:
            logger.error("Insufficient training data")
            return
            
        self.dl_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.dl_model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=2)
        
    def energy_cost(self, path):
        if not path or path[-1] != self.monitor.base_station or len(path) < 2:
            return float('inf')
            
        total = 0
        energy_usage = []
        
        try:
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                distance = self.monitor.topology_map[u][v]['weight']
                traffic = self.monitor.traffic_load[v]
                cost = 0.3 * distance**2 + 0.2 * traffic
                total += cost
                energy_usage.append(self.monitor.node_energy[u])
        except (KeyError, nx.NetworkXError):
            return float('inf')
        
        if not energy_usage:
            return float('inf')
            
        min_energy = min(energy_usage)
        return total + 5 * max(0, 0.2 - min_energy)
    
    def run_routing_cycle(self):
        self.monitor.update_state()
        
        active_nodes = [n for n in self.monitor.node_energy 
                      if n != self.monitor.base_station]
        if not active_nodes:
            return []
        
        dl_input = []
        for node in active_nodes:
            try:
                distance = nx.shortest_path_length(
                    self.monitor.topology_map,
                    node,
                    self.monitor.base_station,
                    weight='weight'
                )
            except nx.NetworkXNoPath:
                distance = self.monitor.area_size * np.sqrt(2)
            
            dl_input.append([
                self.monitor.node_energy[node],
                self.monitor.traffic_load[node],
                distance / (self.monitor.area_size * np.sqrt(2))
            ])
        
        dl_pred = self.dl_model.predict(np.array(dl_input), verbose=0)
        
        pso_params = self.pso.optimize(
            lambda x: self._pso_cost(x, dl_pred, active_nodes)
        )
        
        self.aco.apply_pso_parameters(pso_params)
        paths = self.aco.find_paths(self.monitor.topology_map)
        self.aco.update_pheromones(paths, self.energy_cost)
        return paths

    def _pso_cost(self, params, dl_pred, active_nodes):
        energy_w, traffic_w, distance_w = params
        total = 0
        
        for idx, node in enumerate(active_nodes):
            try:
                next_hop = np.argmax(dl_pred[idx])
                neighbors = list(self.monitor.topology_map.neighbors(node))
                
                if next_hop not in neighbors:
                    total += 1000
                    continue
                
                energy_cost = (1 - self.monitor.node_energy[next_hop]) * energy_w
                traffic_cost = self.monitor.traffic_load[next_hop] * traffic_w
                distance_cost = self.monitor.topology_map[node][next_hop]['weight'] * distance_w
                
                total += energy_cost + traffic_cost + distance_cost
                
            except (KeyError, nx.NetworkXError):
                total += 1000
                
        return total

    def evaluate_performance(self, num_cycles=200):
        metrics = {
            'energy_consumption': [],
            'success_rate': [],
            'latency': [],
            'network_lifetime': 0,
            'energy_history': []
        }

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap('viridis', self.monitor.num_nodes - 1)
        norm = Normalize(vmin=1, vmax=self.monitor.num_nodes-1)
        
        lines = {}
        for node_id in range(1, self.monitor.num_nodes):
            lines[node_id], = ax.plot([], [], 
                                    color=cmap(norm(node_id)), 
                                    alpha=0.7, 
                                    linewidth=1,
                                    label=f'Node {node_id}')
        
        ax.set_title('Real-time Node Energy Levels')
        ax.set_xlabel('Simulation Cycle')
        ax.set_ylabel('Remaining Energy')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, num_cycles)
        ax.set_ylim(0, 1.1)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Node ID')
        cbar.set_ticks(np.linspace(1, self.monitor.num_nodes-1, 
                                 num=min(10, self.monitor.num_nodes-1)))
        cbar.set_ticklabels([int(x) for x in np.linspace(1, self.monitor.num_nodes-1, 
                                                       num=min(10, self.monitor.num_nodes-1))])

        for cycle in range(num_cycles):
            alive_nodes = sum(1 for e in self.monitor.node_energy.values() if e > 0)
            if alive_nodes <= 1:
                break

            paths = self.run_routing_cycle()
            valid_paths = [p for p in paths if p and p[-1] == 0 and len(p) >= 2]

            if valid_paths:
                best_path = min(valid_paths, key=self.energy_cost)
                metrics['energy_consumption'].append(self.energy_cost(best_path))
                metrics['success_rate'].append(1)
                metrics['latency'].append(len(best_path))
            else:
                metrics['energy_consumption'].append(float('inf'))
                metrics['success_rate'].append(0)
                metrics['latency'].append(float('inf'))

            metrics['network_lifetime'] += 1
            metrics['energy_history'].append(dict(self.monitor.node_energy))

            current_cycle = len(metrics['energy_history'])
            for node_id in lines:
                energy_values = [eh.get(node_id, np.nan) for eh in metrics['energy_history']]
                x_data = np.arange(len(energy_values))
                lines[node_id].set_data(x_data, energy_values)

            ax.set_xlim(0, max(10, current_cycle + 1))
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            plt.pause(0.001) 

            if alive_nodes < 0.3 * self.monitor.num_nodes:
                break

        plt.ioff()
        plt.close(fig)
        
        final_metrics = {
            'avg_energy': np.nanmean([e for e in metrics['energy_consumption'] 
                                    if np.isfinite(e)]),
            'success_rate': np.nanmean(metrics['success_rate']),
            'avg_latency': np.nanmean([l for l in metrics['latency'] 
                                     if np.isfinite(l)]),
            'network_lifetime': metrics['network_lifetime'],
            'energy_history': metrics['energy_history']
        }
        return final_metrics


if __name__ == "__main__":
    router = HybridRouter(num_nodes=35)
    
    logger.info("Training hybrid routing model...")
    router.train_dl_model(epochs=40)
    
    logger.info("Running network simulation...")
    results = router.evaluate_performance()
    
    print("\n=== Final Performance Metrics ===")
    print(f"Average Energy Consumption: {results['avg_energy']:.2f} J")
    print(f"Packet Success Rate: {results['success_rate']:.2%}")
    print(f"Average Path Length: {results['avg_latency']:.1f} hops")
    print(f"Network Lifetime: {results['network_lifetime']} cycles")

    plt.figure(figsize=(16, 6))
    
    ax1 = plt.subplot(121)
    pos = router.monitor.node_positions
    nx.draw(router.monitor.topology_map, pos, with_labels=True,
           node_color=[router.monitor.node_energy.get(n, 0) for n in pos],
           cmap='viridis', ax=ax1)
    ax1.set_title("Network Topology with Node Energy Levels")
    
    ax2 = plt.subplot(122)
    cmap = plt.get_cmap('viridis', router.monitor.num_nodes - 1)
    norm = Normalize(vmin=1, vmax=router.monitor.num_nodes-1)

    for node_id in range(1, router.monitor.num_nodes):
        energies = []
        for cycle_data in results['energy_history']:
            if node_id in cycle_data:
                energies.append(cycle_data[node_id])
            else:
                energies.append(np.nan)
        ax2.plot(energies, color=cmap(norm(node_id)), alpha=0.7, linewidth=1)

    ax2.set_title('Node Energy Levels Over Simulation Cycles')
    ax2.set_xlabel('Simulation Cycle')
    ax2.set_ylabel('Remaining Energy')
    ax2.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='Node ID')
    cbar.set_ticks(np.linspace(1, router.monitor.num_nodes-1, num=min(10, router.monitor.num_nodes-1)))
    cbar.set_ticklabels([int(x) for x in np.linspace(1, router.monitor.num_nodes-1, num=min(10, router.monitor.num_nodes-1))])

    plt.tight_layout()
    plt.show()
