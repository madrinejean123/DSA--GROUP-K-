import itertools 
import numpy as np # - numpy: Used for numerical operations and array manipulation.

# Set a fixed random seed for reproducibility.
np.random.seed(42)

# -------------------------------
# Given adjacency matrix for the TSP
# -------------------------------
adj_matrix = [
    [float('inf'), 12, 10, float('inf'), float('inf'), float('inf'), 12],
    [12, float('inf'), 8, 12, float('inf'), float('inf'), float('inf')],
    [10, 8, float('inf'), 11, 3, float('inf'), 9],
    [float('inf'), 12, 11, float('inf'), 11, 10, float('inf')],
    [float('inf'), float('inf'), 3, 11, float('inf'), 6, 7],
    [float('inf'), float('inf'), float('inf'), 10, 6, float('inf'), 9],
    [12, float('inf'), 9, float('inf'), 7, 9, float('inf')]
]
num_cities = len(adj_matrix)

city_names = ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7"]

# -------------------------------
# Step 1: Preprocess the matrix for MDS
# -------------------------------
adj_matrix_np = np.array(adj_matrix, dtype=float)
large_val = 1000.0
adj_matrix_np[adj_matrix_np == float('inf')] = large_val

# -------------------------------
# Step 2: Use MDS to embed cities into 2D space
# -------------------------------
def mds(D, d=2):
    n = D.shape[0]
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D2).dot(H)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    L = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
    V = eigvecs[:, :d]
    return V.dot(L)

coords = mds(adj_matrix_np, d=2)

# -------------------------------
# Step 3: Self-Organizing Map (SOM) for approximating the TSP tour
# -------------------------------
num_neurons = 20
neurons = np.random.rand(num_neurons, 2)
neurons = neurons * (coords.max(axis=0) - coords.min(axis=0)) + coords.min(axis=0)

iterations = 1000
init_learning_rate = 0.8
init_radius = num_neurons / 2
time_constant = iterations / np.log(init_radius)

for t in range(iterations):
    city_idx = np.random.randint(0, coords.shape[0])
    city = coords[city_idx]
    distances = np.linalg.norm(neurons - city, axis=1)
    winner_idx = np.argmin(distances)
    learning_rate = init_learning_rate * np.exp(-t / iterations)
    radius = init_radius * np.exp(-t / time_constant)
    
    for i in range(num_neurons):
        dist_to_winner = min(abs(i - winner_idx), num_neurons - abs(i - winner_idx))
        if dist_to_winner < radius:
            influence = np.exp(-(dist_to_winner ** 2) / (2 * (radius ** 2)))
            neurons[i] += learning_rate * influence * (city - neurons[i])

# -------------------------------
# Step 4: Construct the TSP tour from the trained SOM
# -------------------------------
city_to_neuron = {}
for i, city in enumerate(coords):
    distances = np.linalg.norm(neurons - city, axis=1)
    city_to_neuron[i] = np.argmin(distances)

sorted_cities = sorted(city_to_neuron, key=lambda k: city_to_neuron[k])
start_index = sorted_cities.index(0)
rotated_cities = sorted_cities[start_index:] + sorted_cities[:start_index]
som_route = rotated_cities + [rotated_cities[0]]

total_cost = 0
for i in range(len(som_route) - 1):
    total_cost += adj_matrix[som_route[i]][som_route[i+1]]

# -------------------------------
# Output the results
# -------------------------------
formatted_route = " -> ".join(city_names[i] for i in som_route)
print("SOM TSP Route:", formatted_route)
print("Total Route Cost (SOM):", total_cost)
