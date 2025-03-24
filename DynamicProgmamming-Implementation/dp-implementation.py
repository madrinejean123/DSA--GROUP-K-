# Convert Adjacency List to Distance Matrix 

# Map city names to numerical indices (0 to 6)
city_to_index = {
    "City 1": 0,
    "City 2": 1,
    "City 3": 2,
    "City 4": 3,
    "City 5": 4,
    "City 6": 5,
    "City 7": 6
}

# Initialize the 7x7 distance matrix with infinity
n = 7
dist_matrix = [[float('inf')] * n for _ in range(n)]
for i in range(n):
    dist_matrix[i][i] = 0  # Distance from  city to itself is equal to 0

# Populate the matrix from the adjacency list
graph = {
    "City 1": {"City 2": 12, "City 3": 10, "City 7": 12},
    "City 2": {"City 1": 12, "City 3": 8, "City 4": 12},
    "City 3": {"City 1": 10, "City 2": 8, "City 4": 11, "City 5": 3, "City 7": 12},
    "City 4": {"City 2": 12, "City 3": 11, "City 5": 11, "City 6": 10},
    "City 5": {"City 3": 3, "City 4": 11, "City 6": 6, "City 7": 9},
    "City 6": {"City 4": 10, "City 5": 6, "City 7": 9},
    "City 7": {"City 1": 12, "City 3": 12, "City 5": 9, "City 6": 9}
}

for city, neighbors in graph.items():
    i = city_to_index[city]
    for neighbor, distance in neighbors.items():
        j = city_to_index[neighbor]
        dist_matrix[i][j] = distance
        dist_matrix[j][i] = distance  # Undirected graph

# Dynamic Programming (Held-Karp) Implementation 

def tsp_dp(dist_matrix):
    n = len(dist_matrix)
    memo = {}  # Stores (current city, visited) -> (min distance, next city)
    
    def visit(city, visited):
        if (city, visited) in memo:
            return memo[(city, visited)]
        # Base case: all cities visited, return to start
        if visited == (1 << n) - 1:
            return dist_matrix[city][0], None
        min_dist = float('inf')
        best_next = None
        for next_city in range(n):
            if not (visited & (1 << next_city)) and dist_matrix[city][next_city] != float('inf'):
                new_visited = visited | (1 << next_city)
                d, _ = visit(next_city, new_visited)
                d += dist_matrix[city][next_city]
                if d < min_dist:
                    min_dist = d
                    best_next = next_city
        memo[(city, visited)] = (min_dist, best_next)
        return min_dist, best_next
    
    # Start from City 0 (City 1) with only itself visited
    total_dist, first_step = visit(0, 1 << 0)
    
    # Reconstruct the path
    path = [0]
    current_city = 0
    visited = 1 << 0
    while len(path) < n:  # Ensure all cities are visited
        _, next_city = memo.get((current_city, visited), (None, None))
        if next_city is None:
            break
        path.append(next_city)
        visited |= (1 << next_city)
        current_city = next_city
    path.append(0)  # Return to start
    
    return total_dist, path

# Execute and Print Results 

total_distance, optimal_route = tsp_dp(dist_matrix)

# Convert numerical indices back to city names
index_to_city = {v: k for k, v in city_to_index.items()}
formatted_route = [index_to_city[i] for i in optimal_route]

print("Optimal Route:", " -> ".join(formatted_route))
print("Total Distance:", total_distance)