import numpy as np
import matplotlib.pyplot as plt

# Distance Matrix from your image (Cities indexed from 0)
dist_matrix = np.array([
    [0, 12, 10, np.inf, np.inf, np.inf, 12],  # City 1 (Start)
    [12, 0, 8, 12, np.inf, np.inf, np.inf],   # City 2
    [10, 8, 0, 11, 3, np.inf, np.inf],        # City 3
    [np.inf, 12, 11, 0, 11, 10, np.inf],      # City 4
    [np.inf, np.inf, 3, 11, 0, 6, 7],         # City 5
    [np.inf, np.inf, np.inf, 10, 6, 0, 9],    # City 6
    [12, np.inf, np.inf, np.inf, 7, 9, 0]     # City 7
])

# Replace infinite values with a large number for computation
INF = 1000
dist_matrix[dist_matrix == np.inf] = INF  

# Number of cities
n_cities = len(dist_matrix)

# Initialize weights (neurons) randomly for SOM
weights = np.random.rand(n_cities, 2)

# Learning rate and neighborhood size for SOM
learning_rate = 0.8
sigma = 2.0
iterations = 1000

# Train the SOM
for it in range(iterations):
    # Select a random city
    city_idx = np.random.randint(0, n_cities)
    
    # Find Best Matching Unit (BMU) - closest neuron
    distances = np.linalg.norm(weights - weights[city_idx], axis=1)
    bmu_idx = np.argmin(distances)

    # Update Best Matching Unit and neighbors
    for i in range(n_cities):
        influence = np.exp(-((i - bmu_idx) ** 2) / (2 * (sigma ** 2)))
        weights[i] += influence * learning_rate * (weights[city_idx] - weights[i])

    # Decay learning rate and sigma
    learning_rate *= 0.999
    sigma *= 0.999

# Get optimized order by sorting neurons
route = np.argsort(weights[:, 0])

# Ensure City 1 is the starting and ending point
if route[0] != 0:
    route = np.roll(route, -np.where(route == 0)[0][0])
route = np.append(route, route[0])  # Ensure the route returns to start

# Compute total distance of the route
def compute_total_distance(route, dist_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i], route[i + 1]]
    return total_distance

total_distance = compute_total_distance(route, dist_matrix)

# Convert 0-based indices to 1-based city numbers
route += 1

# Print results
print(f"Optimal Route: {route}")
print(f"Total Distance: {total_distance:.2f}")

# Plot the optimized route
plt.figure(figsize=(8, 5))
plt.plot(route, "bo-", markersize=8)
plt.title("Optimized TSP Route using SOM (City 1 Start & End)")
plt.xlabel("Step")
plt.ylabel("City Number")
plt.grid()
plt.show()
