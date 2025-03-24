import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize Variables
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

cities = [
    City(0, 0),  # City 1
    City(10, 0),  # City 2
    City(5, 5),   # City 3
    City(0, 10),  # City 4
    City(10, 10), # City 5
    City(5, 15),  # City 6
    City(15, 5),  # City 7
]

num_neurons = 20
neurons = np.random.rand(num_neurons, 2)  # Initialize neurons randomly

# Step 2: Train the SOM
def gaussian_distance(neuron, city, sigma):
    return np.exp(-((neuron[0] - city.x) ** 2 + (neuron[1] - city.y) ** 2) / (2 * sigma ** 2))

def update_neurons(neurons, city, learning_rate, sigma):
    for i in range(num_neurons):
        distance = gaussian_distance(neurons[i], city, sigma)
        neurons[i] += learning_rate * distance * np.array([city.x, city.y] - neurons[i])

max_iterations = 1000
learning_rate = 0.1
sigma = 5.0

for iteration in range(max_iterations):
    for city in cities:
        # Find BMU
        bmu_index = np.argmin(np.linalg.norm(neurons - np.array([city.x, city.y]), axis=1))
        bmu = neurons[bmu_index]

        # Update neurons
        update_neurons(neurons, city, learning_rate, sigma)

    # Reduce learning rate and neighborhood size over time
    learning_rate *= 0.99
    sigma *= 0.99

# Step 3: Extract the Optimized Route
def find_bmu(neurons, city):
    return np.argmin(np.linalg.norm(neurons - np.array([city.x, city.y]), axis=1))

route = []
for city in cities:
    bmu_index = find_bmu(neurons, city)
    route.append(bmu_index)

# Sort cities based on their BMU order along the neuron ring
sorted_cities = [cities[i] for i in np.argsort(route)]

# Step 4: Display Results
print("Optimized Route:")
for city in sorted_cities:
    print(f"City {city.x}, {city.y}")

# Plot the graph
plt.scatter([city.x for city in cities], [city.y for city in cities], c='r')
plt.scatter(neurons[:, 0], neurons[:, 1], c='b')
plt.plot([city.x for city in sorted_cities] + [sorted_cities[0].x], [city.y for city in sorted_cities] + [sorted_cities[0].y], c='g')
plt.show()
