import itertools  # Importing itertools for generating subsets of cities efficiently

# Adjacency matrix representing the graph (TSP problem)
# Each value represents the cost/distance between cities
adj_matrix = [
    [float('inf'), 12, 10, float('inf'), float('inf'), float('inf'), 12],
    [12, float('inf'), 8, 12, float('inf'), float('inf'), float('inf')],
    [10, 8, float('inf'), 11, 3, float('inf'), 9],
    [float('inf'), 12, 11, float('inf'), 11, 10, float('inf')],
    [float('inf'), float('inf'), 3, 11, float('inf'), 6, 7],
    [float('inf'), float('inf'), float('inf'), 10, 6, float('inf'), 9],
    [12, float('inf'), 9, float('inf'), 7, 9, float('inf')]
]

num_cities = len(adj_matrix)  # Number of cities in the problem

# Function to solve the TSP problem using the Dynamic Programming approach (Held-Karp algorithm)
def tsp_dynamic_programming():
    dp = {}  # Dictionary to store the minimum cost to reach a subset of cities ending at a particular city
    parent = {}  # Dictionary to store the previous city in the optimal path
    
    # Initializing base cases: Starting from city 0 to all other reachable cities
    for i in range(1, num_cities):
        if adj_matrix[0][i] < float('inf'):  # If there's a valid path from city 0 to city i
            dp[(1 << i, i)] = adj_matrix[0][i]  # Storing cost to reach city i from 0
            parent[(1 << i, i)] = 0  # Remember that city 0 is the previous city
    
    # Iterating over all subsets of cities of increasing sizes
    for subset_size in range(2, num_cities):  # We start from subsets of size 2 up to n-1
        for subset in itertools.combinations(range(1, num_cities), subset_size):  # Generate all possible subsets
            bitmask = sum(1 << i for i in subset)  # Convert subset into a bitmask representation
            for j in subset:  # Iterate over each city in the subset
                prev_bitmask = bitmask & ~(1 << j)  # Remove city j from the subset
                
                # Finding the minimum cost path to reach city j from any other city in the subset
                candidates = [(dp[(prev_bitmask, k)] + adj_matrix[k][j], k) 
                              for k in subset if k != j and (prev_bitmask, k) in dp and adj_matrix[k][j] < float('inf')]
                
                # If there are valid paths, store the best one (minimum cost)
                if candidates:
                    dp[(bitmask, j)], parent[(bitmask, j)] = min(candidates)
    
    # Constructing the optimal TSP path
    final_bitmask = (1 << num_cities) - 2  # Bitmask representing all cities except the starting city (0)
    
    # Finding the minimum cost path back to the starting city (completing the tour)
    candidates = [(dp[(final_bitmask, j)] + adj_matrix[j][0], j) for j in range(1, num_cities) 
                  if (final_bitmask, j) in dp and adj_matrix[j][0] < float('inf')]
    
    if not candidates:
        return [], float('inf')  # No valid solution found
    
    min_cost, last_city = min(candidates)  # Get the optimal cost and the last city in the tour
    
    # Reconstructing the optimal path from parent dictionary
    path = [0]  # Start from city 0
    bitmask = final_bitmask
    while last_city:
        path.append(last_city)
        next_city = parent.get((bitmask, last_city), 0)  # Retrieve the previous city
        bitmask &= ~(1 << last_city)  # Remove last city from the bitmask
        last_city = next_city
    
    path.append(0)  # Return to the starting city to complete the cycle
    return path[::-1], min_cost  # Reverse the path to get the correct order

# Running the exact TSP solution
exact_path, exact_cost = tsp_dynamic_programming()

# Convert numerical indices to human-readable city names
city_names = ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7"]
formatted_path = " -> ".join(city_names[i] for i in exact_path)

# Printing the results
print("Optimal TSP Route:", formatted_path)
print("Total Route Cost:", exact_cost)
