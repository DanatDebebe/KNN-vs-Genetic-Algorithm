
import numpy as np
import matplotlib.pyplot as plt
import random
#ao ∈ [0, 2], a1 ∈ [−2, 0], and a2 ∈ [−1, 1].
def my_fun(x,y,a0,a1,a2):
    return (a0*np.cbrt(x-5)) + (a1*np.cbrt(y+5)) + a2

#computes the mean square error
def obj_function(pred, actVal):
    
    return np.square(np.subtract(pred, actVal)).mean() 

"""
-genrates population given a size
-chromosomes with 3 genes
-values for genes randomly sampled with the respective ranges
-ao ∈ [0, 2], a1 ∈ [−2, 0], and a2 ∈ [−1, 1]

"""
def gen_population(size):
    population = []
    for _ in range(size):
        population.append([random.uniform(0,2.0001), random.uniform(-2,0.0001), random.uniform(-1,1.0001)])
    
    #print("Generated population of size ", size)
    return population


"""
-evaluates the fitness of a chromosome
-gets predicted values f(x,y) for all (x,y) pairs in the traing data
-fitness is the error value measured by obj_fun()

"""

def evaluate(chromosome, dataset):
    predictions = [] 
    for i in range(len(tset)) :
        predictions.append(my_fun(dataset[i][0],dataset[i][1],chromosome[0], chromosome[1], chromosome[2]))
    fitness = obj_function(predictions, dataset[2])
    
    
    return fitness



def roulette_wheel_selection(population):
    
    fitness_values = [1/evaluate(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    
    # Calculate selection probabilities
    selection_probs = [fitness / total_fitness for fitness in fitness_values]
    
    # Perform roulette wheel spin
    cumulative_probability = [i for i in selection_probs]
    for i in range(1,len(population)):
        cumulative_probability[i] = cumulative_probability[i-1] + selection_probs[i]
    
    selected_index = -1
    spin = random.uniform(0, 1)


def select_parents(population,numPairs):
    parents = []
    #generate parent pairs array the half the size of the population
    for _ in range(numPairs):
        parents.append([roulette_wheel_selection(population),roulette_wheel_selection(population)])
    return parents

    

    for i, prob in enumerate(selection_probs):
        if spin <= cumulative_probability[i]:
            selected_index = i
            break

    return population[selected_index]
# Function to perform uniform crossover between two parents
def crossover(parent1, parent2, crossover_probability):
    
    # Create an empty child chromosome
    child1 = [0]*3
    child2 = [0]*3
    crossover = False
    # Uniformly select genes from parents to create the child chromosome
    for i in range(len(parent1)):
        if random.uniform(0,1) <= crossover_probability:
            
            r = random.uniform(0,1)
            
            child1[i] = (parent1[i] * r) + (parent2[i]*(1-r))
            child2[i] = (parent1[i] * (1-r)) + (parent2[i] * r)
            crossover = True
    if crossover:
        #print(f"Crossover beetween {parent1} and {parent2} Resulted in {child1} and {child2}")
        return child1, child2
    else:
        #print(f"No Crossover between {parent1} and {parent2}")
        return parent1, parent2
def mutate(chromosome,mutation_probability):
    mutated = False
    for i in range(len(chromosome)):
        if random.uniform(0,1) <= mutation_probability:
            #print(f"Performing mutation on chromosome {chromosome} at index {i}")
            mutated = True
            
            if i == 0:
                chromosome[i] = random.uniform(0, 2.0001)
            if i == 1:
                chromosome[i] = random.uniform(-2, 0.001)
            if i == 2:
                chromosome[i] = random.uniform(-1, 1.001)
            
            
    #if mutated:
        #print(f"Resulting chromosome: {chromosome}\n")
    #else:
        #print(f"No mutation for {chromosome}\n")
    
    return chromosome
   
def get_best(population,elitism_size):
    fitness_values = []
    elites = []
    for chromosome in population:
        fitness_values.append(evaluate(chromosome))
    average_fitness = sum(fitness_values) / len(population)   
    elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elitism_size]
    elites = [population[i] for i in elite_indices]
    
    return elites[0], elites, average_fitness

""" 
-pop_size: population size
-cprob: crossover probablity
-e_threshold: epsilon value
-elitism_size: number of best chromosomes to preserve to the next generation

Termination stategy: Run till there the best fitness is lower than 2.905
Muation: Non unifiorm muation, based on the change of the fitness between generations
New population: Crossover and Elitism      
"""

def genetic_algorithm_final(dataset):
    #properties of the Genetic Algorithm
    pop_size = 10
    cprob = 0.75
    e_threshold = 0.01
    stagnation_limit = 1
    elitism_size = 3

    # Initialization
    mutation_prob = 0.02
    pop = gen_population(pop_size)
    numPairs = pop_size//2
    best_fitness = 100 
    stagnation_count = 0
    generation = 0
    fitnesses = []
    while(True):
        
        generation +=1
        parent_pairs = select_parents(pop, numPairs)
        current_best_chrom, elites, average_fitness = get_best(pop, elitism_size)
        current_best_fitness = evaluate(current_best_chrom, dataset)
        
        diversity_check = abs((best_fitness-current_best_fitness)/best_fitness)
        
        # Check for stagnation in the best fitness
        
        if diversity_check < e_threshold:
            stagnation_count += 1
        else:
            stagnation_count = 0
        
        if stagnation_count >= stagnation_limit:  
            mutation_prob *= 1.8   
        else:
            mutation_prob *= 0.2  
        new_population = []
        for i in parent_pairs:
            child1, child2 = crossover(np.array(i[0]),np.array(i[1]),crate)
            child1 = mutate(child1,mutation_prob)
            child2 = mutate(child2,mutation_prob)
            new_population.append(child1)
            new_population.append(child2)

        # Update population for the next generation
        pop = elites + new_population
            
        best_fitness = current_best_fitness
      
        fitnesses.append(best_fitness)
        if best_fitness < 2.905 :
            print(f"Terminating on generation {generation} due to good fitness value of {best_fitness}. ")
            print(f"Best chromosome: {current_best_chrom}")
            break
    x = np.arange(generation)
    y = fitnesses
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show
    return current_best_chrom
   

