#ao ∈ [0, 2], a1 ∈ [−2, 0], and a2 ∈ [−1, 1].
def my_fun(x,y,a0,a1,a2):
    return (a0*np.cbrt(x-5)) + a1*np.cbrt(y+5) + a2

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
        population.append([random.uniform(0,2), random.uniform(-2,0), random.uniform(-1,1)])
    
    #print("Generated population of size ", size)
    return population


"""
-evaluates the fitness of a chromosome
-gets predicted values f(x,y) for all (x,y) pairs in the traing data
-fitness is the error value measured by obj_fun()

"""

def evaluate(chromosome):
    predictions = [] 
    for i in range(len(tset)) :
        predictions.append(my_fun(tset[i][0],tset[i][1],chromosome[0], chromosome[1], chromosome[2]))
    fitness = obj_function(predictions, tz)
    
    
    return fitness


pop = gen_population(5)
pop
