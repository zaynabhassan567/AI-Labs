import random

# Define constants
POPULATION_SIZE = 10
CHROMOSOME_LENGTH = 5
MUTATION_RATE = 0.1
GENERATIONS = 20

# Fitness function: f(x) = x^2
def fitness_function(chromosome):
    x = int(chromosome, 2)  # Convert binary to integer
    return x ** 2

# Generate initial population
def initialize_population(size, length):
    return [''.join(random.choice('01') for _ in range(length)) for _ in range(size)]

# Selection: Roulette Wheel
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    cumulative_probs = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    r = random.random()
    for i, cp in enumerate(cumulative_probs):
        if r <= cp:
            return population[i]

# Crossover: Single-Point
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation
def mutate(chromosome):
    mutated = ''.join(
        bit if random.random() > MUTATION_RATE else str(1 - int(bit)) 
        for bit in chromosome
    )
    return mutated

# Main Genetic Algorithm function
def genetic_algorithm():
    # Step 1: Initialize population
    population = initialize_population(POPULATION_SIZE, CHROMOSOME_LENGTH)
    
    for generation in range(GENERATIONS):
        # Step 2: Evaluate fitness
        fitness_scores = [fitness_function(chrom) for chrom in population]
        
        # Print the best solution of the current generation
        best_chromosome = population[fitness_scores.index(max(fitness_scores))]
        best_fitness = max(fitness_scores)
        print(f"Generation {generation}: Best = {best_chromosome} (Fitness = {best_fitness})")
        
        # Step 3: Selection and reproduction
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness_scores)
            parent2 = roulette_wheel_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        # Step 4: Update population
        population = new_population[:POPULATION_SIZE]

    # Final best solution
    final_fitness_scores = [fitness_function(chrom) for chrom in population]
    best_final_chromosome = population[final_fitness_scores.index(max(final_fitness_scores))]
    print(f"\nFinal Best Solution: {best_final_chromosome} (Fitness = {max(final_fitness_scores)})")

# Run the genetic algorithm
genetic_algorithm()
