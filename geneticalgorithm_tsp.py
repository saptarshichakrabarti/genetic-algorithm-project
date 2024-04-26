import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from numba import njit, prange

# Ignore all warnings
warnings.simplefilter("ignore")
	
@njit
def calculate_fitness(population: np.array, distance_matrix: np.array) -> np.array:
	"""
	Calculates the fitness of each individual.
	Parameters:
	- population (np.array): 2D array representing the population of individuals.
	- distance_matrix (np.array): 2D array representing the distance matrix.
	Returns:
	- fitness (np.array): 1D array containing the fitness values for each individual.
	"""

	# Get the shape of the population and distance matrix
	population_size, individual_size = population.shape
	num_locations = distance_matrix.shape[0]

	# Initialize an array to store fitness values
	fitness = np.zeros(population_size)

	# Loop through each individual in the population
	for i in range(population_size):
		individual = population[i]

		# Calculate total distance for the individual
		total_distance = 0

		# Loop through each location in the individual
		for j in range(individual_size - 1):
			from_location = individual[j] % num_locations
			to_location = individual[j + 1] % num_locations

			# Accumulate the distance between consecutive locations
			total_distance += distance_matrix[from_location, to_location]

		# Add the distance from the last location to the first location
		total_distance += distance_matrix[individual[-1] % num_locations, individual[0] % num_locations]

		# Assign the total distance as the fitness value for the individual
		fitness[i] = total_distance

	return fitness

def evaluate_population(population: np.array, distance_matrix: np.array) -> np.array:
	"""
	Evaluates the population and returns an array of arrays.
	Parameters:
	- population (np.array): 2D array representing the population of individuals.
	- distance_matrix (np.array): 2D array representing the distance matrix.
	Returns:
	- evaluation_results (list): A list containing population fitness, mean fitness,
	best fitness, and the best individual.
	"""

	# Calculate fitness for each individual in the population
	population_fitness = calculate_fitness(population, distance_matrix) 

	# Calculate mean and best fitness values
	mean_fitness = np.mean(population_fitness)
	best_fitness = np.min(population_fitness)

	# Find the best individual based on the index of the minimum fitness
	best_individual = population[np.argmin(population_fitness)]

	# Return the evaluation results as a list
	return [population_fitness, mean_fitness, best_fitness, best_individual]

def update_history(population: np.array,
				distance_matrix: np.array,
				population_history: list,
				generation: int) -> list:
	"""
	Updates the population history.
	Parameters:
	- population (np.array): 2D array representing the population of individuals.
	- distance_matrix (np.array): 2D array representing the distance matrix.
	- population_history (list): List containing historical information about the population.
	- generation (int): Current generation.
	Returns:
	- updated_history (list): Updated population history.
	"""

	# Evaluate the current population
	evaluated_population = evaluate_population(population, distance_matrix)

	# Extract relevant information from the evaluated population
	population_fitness = evaluated_population[0]
	mean_fitness = evaluated_population[1]
	best_fitness = evaluated_population[2]
	best_individual = evaluated_population[3]

	# Append the current generation's information to the history
	population_history.append([generation, population, population_fitness,
                               mean_fitness, best_fitness, best_individual])

	return population_history

def plot_fitness(population_history: list) -> None:
	"""Plots the mean and best fitness of the population over each generation"""
	mean_fitness = [population_history[i][3] for i in range(len(population_history))]
	best_fitness = [population_history[i][4] for i in range(len(population_history))]
	num_cities = len(population_history[0][5])
	# add the best fitness and mean fitness as text in the plot below the x axis
	plt.figure(figsize=(10, 6))
	plt.plot(mean_fitness, label='Mean fitness')
	plt.plot(best_fitness, label='Best fitness')
	plt.title('Mean fitness: {} | Best fitness: {}'.format(mean_fitness[-1], best_fitness[-1]))
	plt.suptitle('Mean and best fitness over generations for {} cities'.format(num_cities))	
	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend()
	plt.savefig('fitness_plot_{}.png'.format(num_cities))
	# plt.show()

""" *** Initialisation functions *** """

def random_population(distance_matrix: np.array, population_size: np.int32, random_seed: np.int32 = 42) -> np.array:
	"""
	Creates a population of population_size individuals.
	Parameters:
	- distance_matrix (np.array): 2D array representing the distance matrix.
	- population_size (np.int32): Size of the population.
	- random_seed (np.int32, optional): Seed for the random number generator. Default is 42.
	Returns:
	- population (np.array): 2D array representing the generated population.
	"""

	# Set the random seed for reproducibility
	np.random.seed(random_seed)

	# Generate a random permutation for each individual in the population
	population = np.array([np.random.permutation(len(distance_matrix)) for _ in range(population_size)], dtype=np.int32)
	# population = np.array([np.roll(np.random.permutation(len(distance_matrix) + 1), shift=1) for _ in range(population_size)], dtype=np.int32)

	return population

def greedy_population(distance_matrix: np.array, population_size: np.int32) -> np.array:
	"""
	Creates a population of population_size individuals using a greedy approach.
	Parameters:
	- distance_matrix (np.array): 2D array representing the distance matrix.
	- population_size (np.int32): Size of the population.
	Returns:
	- population (np.array): 2D array representing the generated population.
	"""

	# Get the number of cities
	num_cities = len(distance_matrix)

	# Initialize the population array
	population = np.zeros((population_size, num_cities), dtype=np.int32) 
	# population = np.zeros((population_size, num_cities + 1), dtype=np.int32) # Add an extra element (num_cities + 1) for the starting city to make tour a cycle

	for i in range(population_size):
		# Initialize variables for the current tour and unvisited cities
		tour = np.zeros(num_cities, dtype=np.int64)
		# tour = np.zeros(num_cities + 1, dtype=np.int64) # Add an extra element (num_cities + 1) for the starting city to make tour a cycle
		unvisited_cities = np.arange(num_cities, dtype=np.int64)

		# Randomly choose the starting city
		current_city = np.random.choice(unvisited_cities)
		tour[0] = current_city
		# Ensure the tour ends with the starting city
		# tour[-1] = current_city  # Ending city is the starting city
		unvisited_cities = np.delete(unvisited_cities, np.argwhere(unvisited_cities == current_city))

		# Build the tour greedily
		for j in range(1, num_cities):
			distances_to_unvisited = distance_matrix[current_city, unvisited_cities]
			next_city = unvisited_cities[np.argmin(distances_to_unvisited)]
			tour[j] = next_city
			unvisited_cities = np.delete(unvisited_cities, np.argwhere(unvisited_cities == next_city))
			current_city = next_city

		# Assign the tour to the population
		population[i] = tour

	return population

def greedier_population(distance_matrix: np.array, population_size: np.int32) -> np.array:
    """
    Creates a population of population_size individuals using a greedy approach.
    Parameters:
    - distance_matrix (np.array): 2D array representing the distance matrix.
    - population_size (np.int32): Size of the population.
    Returns:
    - population (np.array): 2D array representing the generated population.
    """

    # Get the number of cities
    num_cities = len(distance_matrix)

    # Initialize the population array
    population = np.zeros((population_size, num_cities), dtype=np.int32)

    # Initialize a list to store used starting cities
    used_starting_cities = []

    for i in range(population_size):
        # Initialize variables for the current tour and unvisited cities
        tour = np.zeros(num_cities, dtype=np.int64)
        unvisited_cities = np.arange(num_cities, dtype=np.int64)

        # Randomly choose the starting city based on increasing order of distances and not repeated
        distances_to_unvisited = np.sum(distance_matrix[:, unvisited_cities], axis=1)
        valid_starting_cities = np.setdiff1d(unvisited_cities, used_starting_cities)
        if len(valid_starting_cities) == 0:
            # Reset the used starting cities if all cities are used
            used_starting_cities = []
            valid_starting_cities = unvisited_cities
        starting_city = np.argmin(distances_to_unvisited[valid_starting_cities])
        starting_city = valid_starting_cities[starting_city]

        tour[0] = starting_city
        unvisited_cities = np.delete(unvisited_cities, np.argwhere(unvisited_cities == starting_city))

        # Build the tour greedily
        current_city = starting_city
        for j in range(1, num_cities):
            distances_to_unvisited = distance_matrix[current_city, unvisited_cities]
            next_city = unvisited_cities[np.argmin(distances_to_unvisited)]
            tour[j] = next_city
            unvisited_cities = np.delete(unvisited_cities, np.argwhere(unvisited_cities == next_city))
            current_city = next_city

        # Update the list of used starting cities
        used_starting_cities.append(starting_city)

        # Assign the tour to the population
        population[i] = tour

    return population

def initiate_population(distance_matrix: np.array, population_size: np.int32,
						random_seed: np.int32, greedy: bool = False,
						greedy_mix_percentage: np.float64 = 0.5) -> np.array:
	"""
	Initializes the population.
	Parameters:
	- distance_matrix (np.array): 2D array representing the distance matrix.
	- population_size (np.int32): Size of the population.
	- random_seed (np.int32): Seed for the random number generator.
	- greedy (bool, optional): If True, generates a mix of greedy and random populations. Default is False.
	- greedy_mix_percentage (np.float64, optional): Percentage of greedy population in the mix. Default is 0.5.
	Returns:
	- population (np.array): 2D array representing the generated population.
	"""

	# Set the random seed for reproducibility
	np.random.seed(random_seed)

	if greedy:
		# Create a greedy population
		if greedy_mix_percentage >= 0.9:
			greedy_pop = greedier_population(distance_matrix, population_size)
		else:
			greedy_pop = greedy_population(distance_matrix, population_size)

		# Create a random population
		random_pop = random_population(distance_matrix, population_size, random_seed)

		# Determine the size of the mixed population
		mix_size = int(population_size * min(1, greedy_mix_percentage))

		# Concatenate the greedy and random populations
		population = np.concatenate((greedy_pop[:mix_size], random_pop[mix_size:]))
	else:
		# Generate a random population
		population = random_population(distance_matrix, population_size, random_seed)

	return population

# def initiate_population(distance_matrix: np.array, population_size: np.int32,
# 						random_seed: np.int32, greedy: bool = False,
# 						greedy_mix_percentage: np.float64 = 0.5) -> np.array:
# 	"""
# 	Initializes the population.
# 	Parameters:
# 	- distance_matrix (np.array): 2D array representing the distance matrix.
# 	- population_size (np.int32): Size of the population.
# 	- random_seed (np.int32): Seed for the random number generator.
# 	- greedy (bool, optional): If True, generates a mix of greedy and random populations. Default is False.
# 	- greedy_mix_percentage (np.float64, optional): Percentage of greedy population in the mix. Default is 0.5.
# 	Returns:
# 	- population (np.array): 2D array representing the generated population.
# 	"""

# 	# Set the random seed for reproducibility
# 	np.random.seed(random_seed)

# 	if greedy:
# 		# Determine the mix percentages
# 		mix_size = int(population_size * greedy_mix_percentage)

# 		# Create a mixed population
# 		greedy_pop = greedy_population(distance_matrix, population_size)
# 		very_greedy_pop = greedier_population(distance_matrix, population_size)

# 		# Concatenate the mixed populations
# 		population = np.concatenate((greedy_pop, very_greedy_pop))

# 		# Remove duplicate individuals
# 		population = np.unique(population, axis=0)

# 		# Calculate fitness for each individual in the population
# 		population_fitness = calculate_fitness(population, distance_matrix)

# 		# Sort the population based on fitness
# 		sorted_indices = np.argsort(population_fitness)
# 		population = population[sorted_indices]

# 		# Shuffle the population
# 		np.random.shuffle(population)

# 		# Create a random population
# 		random_pop = random_population(distance_matrix, population_size, random_seed)

# 		# Concatenate the mixed and random populations
# 		population = np.concatenate((population[:mix_size], random_pop[mix_size:]))

# 		# Select the top individuals based on the population size
# 		population = population[:population_size]
# 	else:
# 		# Generate a random population
# 		population = random_population(distance_matrix, population_size, random_seed)

# 	return population



""" *** Selection function *** """

def k_tournament_selection(population: np.array, 
						population_fitness: np.array, 
						lamda: np.int32, k: np.int64) -> np.array:
	"""
	K-tournament selection.
	Parameters:
	- population (np.array): 2D array representing the population of individuals.
	- population_fitness (np.array): 1D array representing the fitness values of the population.
	- lamda (np.int32): Number of parents to select.
	- k (np.int64): Number of individuals in each tournament.
	Returns:
	- selected_parents (np.array): 2D array containing the selected parents.
	"""

	selected_parents = []
	population_size = len(population)

	# Continue until the desired number of parents (lamda) is selected
	while len(selected_parents) < lamda:
		# Randomly choose k individuals for the tournament without replacement
		tournament_indices = np.random.choice(population_size, k, replace=False)
		
		# Extract the fitness values for the selected individuals
		tournament_fitness = population_fitness[tournament_indices]
		
		# Find the index of the winner (individual with the minimum fitness)
		winner_index = tournament_indices[np.argmin(tournament_fitness)]
		
		# Extract the winner individual
		winner_individual = population[winner_index]

		# Ensure the winner is not already selected
		if not any(np.array_equal(winner_individual, ind) for ind in selected_parents):
			selected_parents.append(winner_individual)

	# Convert the list of selected parents to a NumPy array
	return np.array(selected_parents)

""" *** Crossover functions *** """

@njit
def partially_mapped_crossover(parent1: np.array,
								parent2: np.array,
								crossover_probability: np.float64) -> np.array:
		"""
		Perform Partially-Mapped Crossover (PMX) on two parent chromosomes.

		Parameters:
		- parent1 (numpy.ndarray): First parent chromosome.
		- parent2 (numpy.ndarray): Second parent chromosome.
		- crossover_probability (float): Probability of crossover, defaults to 1.0.

		Returns:
		- offspring1 (numpy.ndarray): First offspring chromosome.
		- offspring2 (numpy.ndarray): Second offspring chromosome.
		"""
		size = len(parent1)

		# Check if crossover should be performed
		if np.random.rand() > crossover_probability:
			return parent1.copy(), parent2.copy()

		crossover_points = np.sort(np.random.choice(size, 2, replace=False))

		start, end = crossover_points
		mapping = np.zeros_like(parent1)

		# Copy the segment between crossover points
		offspring1 = parent1.copy()
		offspring2 = parent2.copy()
		offspring1[start:end] = parent2[start:end]
		offspring2[start:end] = parent1[start:end]

		# Create mappings
		for i in range(start, end):
			mapping[parent2[i]] = parent1[i]
			mapping[parent1[i]] = parent2[i]

		# Apply mapping to the rest of the chromosome
		for i in range(size):
			if start <= i < end:
				continue
			while offspring1[i] in parent2[start:end]:
				index = np.where(parent2[start:end] == offspring1[i])[0][0]
				offspring1[i] = parent1[start + index]
			while offspring2[i] in parent1[start:end]:
				index = np.where(parent1[start:end] == offspring2[i])[0][0]
				offspring2[i] = parent2[start + index]

		return offspring1, offspring2

def two_point_crossover(parent1, parent2, crossover_probability=1.0):
    """
    Perform Two-Point Crossover on two parent chromosomes.

    Parameters:
    - parent1 (numpy.ndarray): First parent chromosome.
    - parent2 (numpy.ndarray): Second parent chromosome.
    - crossover_probability (float): Probability of crossover, defaults to 1.0.

    Returns:
    - offspring1 (numpy.ndarray): First offspring chromosome.
    - offspring2 (numpy.ndarray): Second offspring chromosome.
    """
    size = len(parent1)

    # Check if crossover should be performed
    if np.random.rand() > crossover_probability:
        return parent1.copy(), parent2.copy()

    crossover_points = np.sort(np.random.choice(size, 2, replace=False))

    start, end = crossover_points
    offspring1 = np.zeros(size, dtype=np.int64)
    offspring2 = np.zeros(size, dtype=np.int64)

    # Copy the segment between crossover points
    offspring1[start:end] = parent1[start:end]
    offspring2[start:end] = parent2[start:end]

    idx1 = end
    idx2 = end

    # Fill in the rest of the offspring avoiding repeated elements
    for i in range(size):
        if idx1 == size:
            idx1 = 0
        if idx2 == size:
            idx2 = 0

        if offspring1[idx1] == 0:
            next_chromosome1 = parent2[idx1]
            while next_chromosome1 in offspring1:
                next_chromosome1 = parent2[(idx1 + 1) % size]
            offspring1[idx1] = next_chromosome1

        if offspring2[idx2] == 0:
            next_chromosome2 = parent1[idx2]
            while next_chromosome2 in offspring2:
                next_chromosome2 = parent1[(idx2 + 1) % size]
            offspring2[idx2] = next_chromosome2

        idx1 += 1
        idx2 += 1

    return offspring1, offspring2

@njit
def edge_crossover(parent1: np.array,
                   parent2: np.array,
                   crossover_probability: np.float64) -> np.array:
    """
    Edge crossover (EX).
    Parameters:
    - parent1 (np.array): 1D array representing the first parent.
    - parent2 (np.array): 1D array representing the second parent.
    - crossover_probability (np.float64): Probability of crossover.
    Returns:
    - child (np.array): 1D array representing the child after crossover.
    """

    # Check if crossover should occur based on probability
    if np.random.rand() > crossover_probability:
        return parent1.copy()
    else:
        size = parent1.shape[0]

        # Select two random crossover points
        p1, p2 = np.random.choice(size, size=2, replace=False)

        # Ensure p1 is less than p2
        p1, p2 = min(p1, p2), max(p1, p2)

        # Create a mapping of edges from parent1 to parent2
        edges = {parent1[i]: parent2[i] for i in range(size)}

        # Copy the edge section from parent2 to the child
        child = parent1.copy()
        child[p1:p2] = parent2[p1:p2]

        # Adjust the child by replacing repeated elements
        for i in range(p1, p2):
            if child[i] in child[:p1]:
                child[i] = edges[child[i]]
            elif child[i] in child[p2:]:
                child[i] = edges[child[i]]

        return child
	
def order_crossover(parent1, parent2, crossover_probability):
    """
    Perform Order Crossover (OX) on two parent chromosomes.

    Parameters:
    - parent1 (numpy.ndarray): First parent chromosome.
    - parent2 (numpy.ndarray): Second parent chromosome.
    - crossover_probability (float): Probability of crossover.

    Returns:
    - offspring1 (numpy.ndarray): First offspring chromosome.
    - offspring2 (numpy.ndarray): Second offspring chromosome.
    """
    size = len(parent1)

    # Check if crossover should be performed based on probability
    if np.random.rand() > crossover_probability:
        return parent1.copy(), parent2.copy()

    # Randomly select crossover points
    crossover_points = np.sort(np.random.choice(size, 2, replace=False))

    # Copy the segment between crossover points
    segment = parent1[crossover_points[0]:crossover_points[1] + 1]
    
    # Initialize offspring with the segment
    offspring1 = np.full(size, -1)
    offspring2 = np.full(size, -1)

    offspring1[crossover_points[0]:crossover_points[1] + 1] = segment
    offspring2[crossover_points[0]:crossover_points[1] + 1] = segment

    # Fill remaining positions from the other parent
    idx1 = idx2 = 0

    for i in range(size):
        if offspring1[i] == -1:
            while parent2[idx1] in segment:
                idx1 += 1
            offspring1[i] = parent2[idx1]
            idx1 += 1

        if offspring2[i] == -1:
            while parent1[idx2] in segment:
                idx2 += 1
            offspring2[i] = parent1[idx2]
            idx2 += 1

    return offspring1, offspring2
	
# @njit
def crossover(parents: np.array, crossover_probability: np.float64) -> np.array:
	num_parents = len(parents)
	num_children = num_parents * 2
	num_genes = parents.shape[1]

	children = np.empty((num_children, num_genes), dtype=parents.dtype)

	child_index = 0

    # Loop through pairs of parents
	for i in range(0, num_parents - 1, 2):
		parent1 = parents[i]
		parent2 = parents[i + 1]

		# Perform partially mapped crossover (PMX)
		child1_pmx, child2_pmx = partially_mapped_crossover(parent1, parent2, crossover_probability)
		
		children[child_index] = child1_pmx
		children[child_index + 1] = child2_pmx
		child_index += 2

		# Perform two point crossover (TPX)
		child1_ox, child2_ox = order_crossover(parent1, parent2, crossover_probability)

		children[child_index] = child1_ox
		children[child_index + 1] = child2_ox
		child_index += 2
		
	return children

""" *** Mutation functions *** """
@njit
def swap_mutation(individual: np.array,
				mutation_probability: np.float64) -> np.array:
	"""
	Swap mutation.
	Parameters:
	- individual (np.array): 1D array representing the individual.
	- mutation_probability (np.float64): Probability of mutation.
	Returns:
	- mutated_individual (np.array): 1D array representing the mutated individual.
	"""

	# Check if mutation should occur based on probability
	if np.random.rand() > mutation_probability:
		return individual
	else:
		size = individual.shape[0]

		# Select two distinct positions for mutation
		p1, p2 = np.random.choice(size, size=2, replace=False)

		# Perform swap mutation
		individual[p1], individual[p2] = individual[p2], individual[p1]

		return individual
@njit
def inverse_mutation(individual: np.array,
					mutation_probability: np.float64) -> np.array:
	"""
	Inversion mutation.
	Parameters:
	- individual (np.array): 1D array representing the individual.
	- mutation_probability (np.float64): Probability of mutation.
	Returns:
	- mutated_individual (np.array): 1D array representing the mutated individual.
	"""

	# Check if mutation should occur based on probability
	if np.random.rand() > mutation_probability:
		return individual
	else:
		size = individual.shape[0]

		# Select two distinct positions for mutation
		p1, p2 = np.random.choice(size, size=2, replace=False)

		# Ensure p1 is less than p2
		p1, p2 = min(p1, p2), max(p1, p2)

		# Perform inversion mutation
		inverted = individual[p1:p2]
		individual[p1:p2] = inverted[::-1]

		return individual

@njit
def mutate(population, population_percentage, mutation_probability, distance_matrix):
    # Calculate fitness for each individual in the population
    population_fitness = calculate_fitness(population, distance_matrix)

    # Calculate mean fitness value
    mean_fitness = np.mean(population_fitness)

    # Sort the population based on fitness
    sorted_indices = np.argsort(population_fitness)
    sorted_population = population[sorted_indices]

    # Select mutation candidates from the top of the sorted population
    num_mutate = int(len(sorted_population) * population_percentage)
    mutation_candidates = sorted_population[:num_mutate]

    # Initialize an array to store mutated children
    mutated_children = np.empty((num_mutate * 2, population.shape[1]), dtype=population.dtype)

    # Iterate over mutation candidates
    for i in range(num_mutate):
        candidate_index = sorted_indices[i]
        candidate = population[candidate_index]

        # Perform inversion mutation and add to the array
        mutated_children[i * 2] = inverse_mutation(candidate.copy(), mutation_probability)

        # Perform swap mutation and add to the array
        mutated_children[i * 2 + 1] = swap_mutation(candidate.copy(), mutation_probability)

    return np.vstack((population, mutated_children))


""" *** Elimination function *** """

def eliminate_lamda_plus_mu(lamda_population: np.array,
							mu_population: np.array,
							distance_matrix: np.array) -> np.array:
	"""
	Eliminate individuals from the combined population of (lambda + mu) based on fitness.
	Parameters:
	- lamda_population (np.array): 2D array representing the lambda population of individuals.
	- mu_population (np.array): 2D array representing the mu population of individuals.
	- distance_matrix (np.array): 2D array representing the distance matrix.
	Returns:
	- selected_population (np.array): 2D array representing the selected individuals after elimination.
	"""

	# Combine lambda and mu populations
	population = np.vstack((lamda_population, mu_population))

	# Remove duplicate individuals
	population = np.unique(population, axis=0)

	# Combine fitness values
	fitness = calculate_fitness(population, distance_matrix)

	# Sort indices based on fitness
	sorted_indices = np.argsort(fitness)

	# Sort the combined population based on fitness
	sorted_population = population[sorted_indices]

	# Select the top individuals based on the lambda population size
	selected_population = sorted_population[:len(lamda_population)]

	return selected_population


""" *** Local search function *** """

@njit
def k_opt_individual(individual, k):
    new_individual = individual.copy()
    tour_length = len(new_individual)

    for _ in range(k):
        # Select 2 random points in the tour
        a, b = np.random.choice(len(new_individual), size=2, replace=False)
        a, b = min(a, b), max(a, b)
        # Reverse the tour between the 2 points
        if a < b:
            new_individual[a:b] = np.roll(new_individual[a:b], 1)[::-1]
        else:
            # Handle wrap-around for indices
            indices = np.arange(b - 1, a - 1, -1) % tour_length
            new_individual[a:b] = np.roll(new_individual[indices], 1)[::-1]

    return new_individual

@njit(parallel=True)
def k_opt(population, k):
    num_individuals, _ = population.shape
    optimized_population = np.empty_like(population)

    for i in prange(num_individuals):
        optimized_population[i] = k_opt_individual(population[i], k)

    return optimized_population

# 	return optimized_population

""" *** Diversity promotion *** """

def fitness_sharing(population: np.array, 
					population_fitness: np.array, 
					sigma: np.float16 = 1.0, 
					alpha: np.float16 = 2.0) -> np.array:
	"""
	Apply fitness sharing to adjust the fitness values of individuals.
	Args:
	- population: 2D array representing the population of individuals.
	- population_fitness: 1D array representing the fitness values of the population.
	- sigma: Sharing distance.
	- alpha: Shape parameter.
	Returns:
	- adjusted_fitness: 1D array representing the adjusted fitness values.
	"""
	population_size = len(population)
	adjusted_fitness = np.zeros_like(population_fitness)

	def calculate_distance(individual1, individual2, population_fitness):
		"""Calculate Euclidean distance between two individuals."""
		fitness1 = population_fitness[np.where((population == individual1).all(axis=1))[0][0]]
		fitness2 = population_fitness[np.where((population == individual2).all(axis=1))[0][0]]
		return np.linalg.norm(abs(fitness1 - fitness2))
	
	@njit
	def sharing_function(distance, sigma, alpha):
		"""
		Compute sharing value based on distance using a sharing function.
		Args:
		- distance: Distance between individuals.
		- sigma: Sharing distance.
		- alpha: Shape parameter.
		Returns:
		- Sharing value.
		"""
		if distance <= sigma:
			if distance == 0:
				return 1  # Avoid division by zero
			else:
				return (1 - (distance / sigma) ** alpha)
		else:
			return 0

	for i in range(population_size):
		for j in range(population_size):
			if i != j:
				distance_ij = calculate_distance(population[i], population[j], population_fitness)
				sharing_ij = sharing_function(distance_ij, sigma, alpha)
				adjusted_fitness[i] += population_fitness[i] / sharing_ij

	return adjusted_fitness

def check_population_validity(population):
    population_size, individual_size = population.shape

    for i in range(population_size):
        individual = population[i]

        # Check for repeated elements
        if len(set(individual)) < individual_size:
            print(f"Individual {i + 1} has repeated elements.")

        # Check if all elements are in the range of individual_size
        if not np.all(np.isin(individual, np.arange(individual_size))):
            print(f"Individual {i + 1} contains elements outside the valid range.")

    print("Population validity check complete.")

""" *** Main function *** """

def genetic_algorithm(distance_matrix: np.array) -> list:
# def genetic_algorithm(distance_matrix: np.array, parameters: list) -> list:
	"""Genetic algorithm"""

	start_time = time.time()
	max_duration = 300

	# Parameters
	population_size = 120 # Population size or 150
	lamda = 40  # Number of parents
	gen_skip = 2  # Number of generations to skip before applying diversity promotion
	iterations = 2000 # Maximum number of iterations

	num_cities = len(distance_matrix)

	greedy = True  # Greedy population

	# scaled_value = (value - min_from) * (max_to - min_to) / (max_from - min_from) + min_to
	# --> 0.7 to 1.0 instead of 0.5 to 1.0 to favour greeedier solution mix
	greedy_mix_percentage = 0.5 if num_cities < 100 else (num_cities - 50) * (1.0 - 0.7) / (1000 - 50) + 0.7
	
	crossover_probability = 0.95
	initial_mutation_probability = 0.95  # Initial mutation probability
	mutation_population_percentage = 0.90  # Percentage of population to mutate
	random_seed = 42
	k = 4  # Number of individuals in tournament

	sigma=1.2
	alpha=2.5 # previously better at 2.0

	# Initialize mutation probability
	mutation_probability = initial_mutation_probability

	# Initialise population
	population = initiate_population(distance_matrix, population_size, random_seed, greedy, greedy_mix_percentage)
	# check_population_validity(population)
	population_fitness = calculate_fitness(population, distance_matrix)

	# Evaluate population
	generation = 0
	population_history = []
	# population_history = update_history(population, distance_matrix, population_history, generation)
	
	# Main loop
	
	while time.time() - start_time < max_duration and generation < iterations:

		generation += 1
	
		if generation % gen_skip == 0:

			# Update mutation probability
			# mutation_probability = mutation_probability * 0.9999

			# # Update crossover probability
			# crossover_probability = crossover_probability * 0.9999

			# Diversity promotion
			adjusted_fitness = fitness_sharing(population, population_fitness, sigma, alpha)
			population_fitness = adjusted_fitness


		# Selection
		selected_parents = k_tournament_selection(population, population_fitness, lamda, k)
		# check_population_validity(selected_parents)

		# Crossover
		children = crossover(selected_parents, crossover_probability)
		# check_population_validity(children)

		# Mutation
		children = mutate(children, mutation_population_percentage, mutation_probability, distance_matrix)
		# check_population_validity(children)

		# Local search
		if generation % (gen_skip + 1) == 0:
			children = k_opt(children, k=3)

		# check_population_validity(children)

		# Elimination
		population = eliminate_lamda_plus_mu(population, children, distance_matrix)
		
		population_fitness = calculate_fitness(population, distance_matrix)

		# Update population history
		population_history = update_history(population, distance_matrix, population_history, generation)

		print("Generation : ", generation, 
			" | Best fitness : ", population_history[-1][4], 
			" | Mean fitness : ", population_history[-1][3], 
			" | Population shape : ", population.shape)
		
		if generation > 100 and population_history[-1][3] == population_history[-100][3]:
			break

	return population_history
	
file = open('tour50.csv')
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()

best_fit = []
mean_fit = []


pop = genetic_algorithm(distanceMatrix)
best_fit.append(pop[-1][4])
mean_fit.append(pop[-1][3])
save_path = 'fitness_{}.csv'.format(len(distanceMatrix))
# save population history to csv
df = pd.DataFrame(pop, columns=['generation', 'population', 'population_fitness', 'mean_fitness', 'best_fitness', 'best_individual'])
df.to_csv(save_path, index=False)
# save fitness plot
plot_fitness(pop)