# =============================================================================
# AtomTreeSearch: A Quantum-Enhanced Monte Carlo Tree Search Framework
# =============================================================================
# Reference:
# - Title: Quantum-enhanced Monte Carlo Tree Search framework for combinatorial optimization problems
# - Authors: Yohan Finet, Yves Bérubé-Lauzière, Victor Drouin-Touchette
# - arXiv Reference: https://arxiv.org/abs/2606.30415
# =============================================================================

import math
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pulp
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. CORE ALGORITHM IMPLEMENTATION (CORRECTED)
# =============================================================================

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self, legal_actions):
        if self.untried_actions is None:
            self.untried_actions = list(legal_actions)
        return len(self.untried_actions) == 0

class QuantumEnhancedMCTS:
    def __init__(self, exploration_constant=0.6):
        self.cp = exploration_constant
        self.best_state = None
        self.best_value = -float('inf')

    def is_terminal(self, state):
        return False

    def apply_action(self, state, action):
        return state + (action,)

    def select(self, node, get_legal_actions_fn):
        # Selection only traverses when fully expanded
        while not self.is_terminal(node.state):
            actions = get_legal_actions_fn(node.state)
            if not node.is_fully_expanded(actions):
                return node
            node = self.best_child(node)
        return node

    def best_child(self, node):
        log_n_parent = math.log(node.visits)
        weights = [
            (c.total_value / c.visits) + self.cp * math.sqrt((2 * log_n_parent / c.visits))
            for c in node.children
        ]
        return node.children[np.argmax(weights)]

    def expand(self, node, legal_actions):
        action = node.untried_actions.pop()
        next_state = self.apply_action(node.state, action)
        child = MCTSNode(next_state, parent=node, action=action)
        node.children.append(child)
        return child

    def backpropagate(self, node, reward):
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += reward
            curr = curr.parent

    def run(self, initial_state, budget, get_legal_actions_fn):
        root = MCTSNode(initial_state)
        for _ in range(budget):
            v = self.select(root, get_legal_actions_fn)
            actions = get_legal_actions_fn(v.state)
            if not self.is_terminal(v.state) and not v.is_fully_expanded(actions):
                v = self.expand(v, actions)
            reward = self.simulate(v.state)
            self.backpropagate(v, reward)
        return self.best_state

    def run_with_restarts(self, initial_state, total_budget, num_restarts=3, get_legal_actions_fn=None):
        budget_per = total_budget // num_restarts
        for _ in range(num_restarts):
            self.run(initial_state, budget_per, get_legal_actions_fn)
        return self.best_state


# =============================================================================
# 2. TESTCASE 1: MAX-CUT QUBO WITH QISKIT
# =============================================================================

edges = [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)]
Q_matrix = np.array([
    [ 2, -1, -1,  0,  0],
    [-1,  2, -1, -1,  0],
    [-1, -1,  2, -1, -1],
    [ 0, -1, -1,  2, -1],
    [ 0,  0, -1, -1,  2]
])

class QUBO_Qiskit_MCTS(QuantumEnhancedMCTS):
    def __init__(self):
        super().__init__(exploration_constant=0.6)
        self.sampler = SamplerV2()

    def is_terminal(self, state): return len(state) == 5
    def get_legal_actions(self, state): return [0, 1] if len(state) < 5 else []

    def local_search(self, x):
        best_x = list(x)
        best_val = np.array(best_x) @ Q_matrix @ np.array(best_x)
        improved = True
        while improved:
            improved = False
            for i in range(5):
                neighbor = list(best_x)
                neighbor[i] = 1 - neighbor[i]
                val = np.array(neighbor) @ Q_matrix @ np.array(neighbor)
                if val > best_val:
                    best_val = val
                    best_x = neighbor
                    improved = True
        return tuple(best_x), best_val

    def generate_quantum_rollout(self, partial_state):
        k = len(partial_state)
        rem = 5 - k
        if rem == 0: return partial_state
        qc = QuantumCircuit(rem)
        
        # Ry rotation on |0> state for real bias
        for i in range(rem):
            bias = Q_matrix[k+i, k+i]
            theta = np.pi/2 + np.clip(bias, -np.pi/2, np.pi/2)
            qc.ry(float(theta), i)
        qc.measure_all()
        
        job = self.sampler.run([qc], shots=1)
        bitstring = list(job.result()[0].data.meas.get_counts().keys())[0][::-1]
        completion = [int(b) for b in bitstring]
        return partial_state + tuple(completion)

    def simulate(self, state):
        raw_sol = self.generate_quantum_rollout(state)
        refined_sol, val = self.local_search(raw_sol)
        if val > self.best_value:
            self.best_value = val
            self.best_state = refined_sol
        return math.exp(val / (1 + abs(val)))


# =============================================================================
# 3. TESTCASE 2: TRAVELING SALESPERSON PROBLEM (TSP)
# =============================================================================

class TSP_QuantumMCTS(QuantumEnhancedMCTS):
    def __init__(self, num_cities, dist_matrix, exploration_constant=0.6):
        super().__init__(exploration_constant)
        self.num_cities = num_cities
        self.dist_matrix = dist_matrix
        self.min_d = 999.9

    def is_terminal(self, state): return len(state) == self.num_cities
    def get_legal_actions(self, state): return [c for c in range(self.num_cities) if c not in state]

    def calculate_distance(self, path):
        return sum(self.dist_matrix[path[i], path[i+1]] for i in range(len(path)-1)) + self.dist_matrix[path[-1], path[0]]

    def two_opt(self, path):
        best_path = list(path)
        best_dist = self.calculate_distance(best_path)
        improved = True
        while improved:
            improved = False
            for i in range(len(best_path)):
                for j in range(i + 2, len(best_path)):
                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                    new_dist = self.calculate_distance(new_path)
                    if new_dist < best_dist - 1e-6:
                        best_path = new_path
                        best_dist = new_dist
                        improved = True
        return tuple(best_path), best_dist

    def simulate(self, state):
        current_path = list(state)
        unvisited = [c for c in range(self.num_cities) if c not in current_path]
        path = current_path[:]
        while unvisited:
            last = path[-1] if path else random.choice(range(self.num_cities))
            unvisited.sort(key=lambda x: self.dist_matrix[last, x])
            path.append(unvisited.pop(0))
        
        refined_path, d = self.two_opt(path)
        if d < self.min_d:
            self.min_d = d
            self.best_state = refined_path
            self.best_value = -d
        return math.exp(-20.0 * d)


# =============================================================================
# HELPER CLASSICAL SOLVERS
# =============================================================================

def solve_tsp_lp(n, dist_matrix, timeout=30):
    prob = pulp.LpProblem(f"TSP_{n}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", range(n), 0, n - 1, pulp.LpContinuous)
    prob += pulp.lpSum(dist_matrix[i][j] * x[i][j] for i in range(n) for j in range(n))
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1
        prob += pulp.lpSum(x[j][i] for j in range(n) if i != j) == 1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n * x[i][j]) <= n - 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))
    return pulp.value(prob.objective)

def solve_tsp_sa(n, dist_matrix, steps=30000):
    def get_dist(path):
        return sum(dist_matrix[path[i], path[i+1]] for i in range(len(path)-1)) + dist_matrix[path[-1], path[0]]
    curr_path = list(range(n))
    random.shuffle(curr_path)
    curr_dist = get_dist(curr_path)
    best_path, best_dist = curr_path[:], curr_dist
    T = 1.0
    alpha = 0.999
    for _ in range(steps):
        T *= alpha
        new_path = curr_path[:]
        l, r = sorted(random.sample(range(n), 2))
        new_path[l:r] = reversed(new_path[l:r])
        new_dist = get_dist(new_path)
        if new_dist < curr_dist or random.random() < math.exp((curr_dist - new_dist) / (T + 1e-8)):
            curr_path, curr_dist = new_path, new_dist
            if curr_dist < best_dist:
                best_dist, best_path = curr_dist, curr_path
    return best_dist


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

if __name__ == '__main__':
    print("====================================================")
    print("Starting Hybrid Q-MCTS Optimization Benchmarks")
    print("====================================================\n")

    # 1. Run Max-Cut QUBO
    print("--- Testcase 1: Max-Cut QUBO ---")
    engine = QUBO_Qiskit_MCTS()
    engine.run_with_restarts((), 300, 3, engine.get_legal_actions)
    print(f"Optimal Solution Found: {engine.best_state}")
    print(f"Max Cut Value:          {engine.best_value}\n")

    # Plot Max-Cut partition
    fig, ax = plt.subplots(figsize=(6, 4))
    G_qubo = nx.Graph()
    G_qubo.add_edges_from(edges)
    colors = ['#00FFFF' if engine.best_state[i] == 1 else '#FFA500' for i in range(5)]
    nx.draw(G_qubo, with_labels=True, node_color=colors, node_size=800, font_weight='bold', ax=ax)
    ax.set_title("Max-Cut Graph Partition")
    plt.savefig('maxcut_partition_result.png')
    print("Max-Cut partition plot saved as 'maxcut_partition_result.png'\n")

    # 2. Run TSP Scaling
    print("--- Testcase 2: TSP Routing Benchmarks ---")
    tsp_results = []
    
    for n, seed, budget, restarts in [(10, 42, 1500, 3), (19, 61, 2000, 4), (25, 25, 5000, 5)]:
        print(f"Running benchmark for N = {n} Cities...")
        np.random.seed(seed)
        random.seed(seed)
        cities = np.random.rand(n, 2)
        dist_matrix = np.sqrt(np.sum((cities[:, np.newaxis, :] - cities[np.newaxis, :, :])**2, axis=-1))
        
        t0 = time.time()
        lp_dist = solve_tsp_lp(n, dist_matrix)
        t_lp = time.time() - t0
        
        t0 = time.time()
        sa_dist = solve_tsp_sa(n, dist_matrix, steps=budget*20)
        t_sa = time.time() - t0
        
        t0 = time.time()
        mcts = TSP_QuantumMCTS(n, dist_matrix)
        mcts.run_with_restarts((), budget, restarts, mcts.get_legal_actions)
        t_mcts = time.time() - t0
        
        tsp_results.append({
            "N": n,
            "LP Distance": f"{lp_dist:.4f}" if lp_dist is not None else "Timed Out",
            "LP Time": f"{t_lp:.2f}s",
            "SA Distance": f"{sa_dist:.4f}",
            "SA Time": f"{t_sa:.2f}s",
            "MCTS Distance": f"{mcts.min_d:.4f}",
            "MCTS Time": f"{t_mcts:.2f}s"
        })

    # Print summary table
    df = pd.DataFrame(tsp_results)
    print("\n=== Benchmarking Summary Table ===")
    print(df.to_string(index=False))
    print("\nAll experiments finished successfully!")
