import itertools  

# Adjacency matrix representing the graph (TSP problem)
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

def tsp_dynamic_programming():
    dp = {}  # Stores the minimum cost to reach a subset of cities ending at a particular city
    parent = {}  # Stores the previous city in the optimal path
    
    # Initialize base cases
    for i in range(1, num_cities):
        if adj_matrix[0][i] < float('inf'):
            dp[(1 << i, i)] = adj_matrix[0][i]  # Cost to reach city i from 0
            parent[(1 << i, i)] = 0  # Previous city is 0
    
    # Iterate over subsets of cities of increasing sizes
    for subset_size in range(2, num_cities):
        for subset in itertools.combinations(range(1, num_cities), subset_size):
            bitmask = sum(1 << i for i in subset)
            for j in subset:
                prev_bitmask = bitmask & ~(1 << j)
                candidates = [(dp[(prev_bitmask, k)] + adj_matrix[k][j], k) 
                              for k in subset if k != j and (prev_bitmask, k) in dp and adj_matrix[k][j] < float('inf')]
                if candidates:
                    dp[(bitmask, j)], parent[(bitmask, j)] = min(candidates)
    
    # Find the minimum cost path back to the starting city
    final_bitmask = (1 << num_cities) - 2
    candidates = [(dp[(final_bitmask, j)] + adj_matrix[j][0], j) for j in range(1, num_cities) 
                  if (final_bitmask, j) in dp and adj_matrix[j][0] < float('inf')]
    
    if not candidates:
        return [], float('inf')
    
    min_cost, last_city = min(candidates)
    
    # Reconstruct the optimal path
    path = [0]
    bitmask = final_bitmask
    while last_city:
        path.append(last_city)
        next_city = parent.get((bitmask, last_city), 0)
        bitmask &= ~(1 << last_city)
        last_city = next_city
    
    path.append(0)  # Return to the starting city
    return path[::-1], min_cost

exact_path, exact_cost = tsp_dynamic_programming()

# Convert numerical indices to human-readable city names
city_names = ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7"]
formatted_path = " -> ".join(city_names[i] for i in exact_path)

# Print the results
print("Optimal TSP Route:", formatted_path)
print("Total Route Cost:", exact_cost)
