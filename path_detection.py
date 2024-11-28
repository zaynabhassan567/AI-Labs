import math
import random
import matplotlib.pyplot as plt

# Pre-defined cities with coordinates (City Name, x, y)
CITIES = [
    ("Depot", 0, 0),
    ("City A", 2, 4),
    ("City B", 5, 2),
    ("City C", 7, 8),
    ("City D", 1, 7),
    ("City E", 4, 9)
]

# Calculate Euclidean distance between two cities
def calculate_distance(city1, city2):
    x1, y1 = city1[1], city1[2]
    x2, y2 = city2[1], city2[2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Compute total route distance
def total_route_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])
    distance += calculate_distance(route[-1], route[0])  # Return to start
    return distance

# Generate initial population
def initialize_population(size, cities):
    population = []
    for _ in range(size):
        route = random.sample(cities, len(cities))  # Random permutation of cities
        population.append(route)
    return population

# Fitness function: Inverse of route distance
def fitness_function(route):
    return 1 / total_route_distance(route)

# Selection: Tournament selection
def tournament_selection(population, fitness_scores, k=3):
    selected = random.sample(list(zip(population, fitness_scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)  # Higher fitness is better
    return selected[0][0]

# Crossover: Order Crossover (OX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copy a segment from parent1
    child[start:end] = parent1[start:end]
    # Fill remaining positions with parent2's genes in order
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

# Mutation: Swap Mutation
def mutate(route, mutation_rate=0.2):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

# Genetic Algorithm
def genetic_algorithm(cities, population_size=50, generations=100, mutation_rate=0.1):
    # Step 1: Initialize population
    population = initialize_population(population_size, cities)
    
    for generation in range(generations):
        # Step 2: Evaluate fitness
        fitness_scores = [fitness_function(route) for route in population]
        
        # Track the best solution
        best_idx = fitness_scores.index(max(fitness_scores))
        best_route = population[best_idx]
        best_distance = total_route_distance(best_route)
        
        print(f"Generation {generation}: Best Distance = {best_distance:.2f}")
        
        # Step 3: Selection and reproduction
        new_population = []
        for _ in range(population_size // 2):  # Half the population as parents
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        
        # Step 4: Update population
        population = new_population
    
    # Final best route
    final_fitness_scores = [fitness_function(route) for route in population]
    best_idx = final_fitness_scores.index(max(final_fitness_scores))
    best_route = population[best_idx]
    best_distance = total_route_distance(best_route)
    
    print(f"\nFinal Best Route: {[city[0] for city in best_route]}")
    print(f"Final Best Distance: {best_distance:.2f}")
    return best_route

# Visualize the route
def visualize_route(route):
    x = [city[1] for city in route] + [route[0][1]]
    y = [city[2] for city in route] + [route[0][2]]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    for i, city in enumerate(route):
        plt.text(city[1], city[2], city[0], fontsize=12)
    plt.title("Optimized Delivery Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Run the Genetic Algorithm for Delivery Route Optimization
optimized_route = genetic_algorithm(CITIES, generations=50)
visualize_route(optimized_route)
