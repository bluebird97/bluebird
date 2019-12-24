#encoding: utf-8
"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt
import array

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds
a = 0.3                  # invert coefficient

def F(x): 
	retArr = []
	for ele in x:
		retArr.append(np.sin(10*ele)*ele + np.cos(2*ele)*ele)
	return  np.asarray(retArr)     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

def invert_R1(pop,fitness): #
    S_total = []
    R_total = []
    r = 0
    fitness_aver = fitness.sum()/POP_SIZE
    print("the average of fitness:",fitness_aver)
    for index in range(len(fitness)):
        if fitness[index] >= fitness_aver:                    
            S_total.append(translateDNA(pop[index]))
        else:
            R_total.append(translateDNA(pop[index]))
        
    S = S_total.copy()                          
    R = R_total.copy()
    r = abs(len(S_total) - len(R_total))   
    print("r =", r)
    if len(S_total) < len(R_total):           
        for i in range (r):
            del R_total[np.random.choice(range(len(R_total)))]
    else: # tt fix , we don't use another if to judge
        for i in range (r):
            del S_total[np.random.choice(range(len(S_total)))]
    # now we got the same size length list   
    print("len(S_total): ", len(S_total))
    print("len(R_total): ", len(R_total))

    R1_invert = [(1-a) * S_total[i]+ a * R_total[i] for i in range(len(S_total))] 
    print("===============================")
    print(len(R1_invert), "<<<<R1_invert.len")
    print("R1_invert =",R1_invert)
    print("===============================")

    R1 = R1_invert + S

    print("")
    print(len(R1), "<<<<R1.len")  

    print("===============================")
    print("R1 =",R1)
    print("===============================")          
    R1_F = F(R1)                      
    #R1_fitness =  get_fitness(R1_F)
    #print("R1 =",R1) ## tt1
    #print("R1_fitness =", R1_fitness)
    #while R1_fitness < fitness_aver:
        
        #R_total = R1
        #R1 = [(1-a)*S_total[i]+a*R_total[i] for i in range(len(S_total))]
    #print("R1 =", R1)
    #print("R1_invert's length:", len(R1_invert))
    #print("R1's length:", len(R1))
    return R1

def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

T = translateDNA(pop)
F_values = F(T)
fitness = get_fitness(F_values)

R1 = invert_R1(pop,fitness)
#print("S_total's min:",np.argmin(get_fitness(S)))
print("max_fitness:",np.argmax(fitness))
print("fitness:",fitness)
print("F_values:",F_values)
print("T's value :", T)
#print("all DNA:", pop)
#print("R1:", R1)

