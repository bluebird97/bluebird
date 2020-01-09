import numpy as np
import matplotlib.pyplot as plt
import math

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds
a = 0.3                  # invert coefficient

def F(transpop):
    F = [np.sin(10*transpop[i])*transpop[i] + np.cos(2*transpop[i])*transpop[i] for i in range(len(transpop))]
    return F

def get_fitness(pred):
    fit = [pred[i] + 1e-3 - np.min(pred) for i in range(len(pred))]
    return fit

def initial_origin(POP_SIZE,DNA_SIZE):
    pop = []
    for i in range(POP_SIZE):
        temporary = []
        for j in range(DNA_SIZE):
            temporary.append(np.random.randint(0,2))
        pop.append(temporary)
    return pop


#从二进制到十进制
 #input:种群,染色体长度

def transDNA(pop):
    temp = [] # temp == transedDNA
    a = 0
    for i in range(POP_SIZE):
        a = sum(pop[i][j]*(math.pow(2,j))for j in range(DNA_SIZE))/float(2**DNA_SIZE-1) * 5 
        #for j in range(DNA_SIZE):
            
        temp.append(a) # temp 是一个 一维 列表，like[3,4,5,....]
    return temp
        
def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


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


def invert_R1(pop,fitness):
    fitness_aver = sum(fitness)/POP_SIZE
    S_total = []
    R_total = []
    R1_invert = []
    r = 0
    print("the average of fitness:",fitness_aver)
    for index in range(len(fitness)):
        if fitness[index] >= fitness_aver:
            # S 是强壮的群体
            S_total.append(pop[index])

        else:
            # R 是弱势群体
            R_total.append(pop[index])
            
    S = S_total.copy()
    R = R_total.copy()
    r = abs(len(S_total) - len(R_total))

    # 修剪 R S 这两个列表，令其二者长度 为 相同 的短度
    # 如果 S 短，删除R的
    if len(S_total) < len(R_total):
        print("R longer, can happen")
        for i in range (r):
            del R_total[np.random.choice(range(len(R_total)))]
    # 如果S 长， 删除S的
    else:
        print("S longer, can happen, too")
        for i in range (r):
            del S_total[np.random.choice(range(len(S_total)))]

    print("tt10 S_total == ", S_total)
    print("tt11 R_total == ", R_total)
    for i in range(len(S_total)):
        print("in tt3 for")
        # 这句话的意思是： 让S和R交叉 生成初始的R1——invert
        R1_pre = [(1-a)*S_total[i][j]+a*R_total[i][j] for j in range(DNA_SIZE)]
        # R1_pre 的意义是： R1_pre 暂存数据for R1_invert
        for k in range(DNA_SIZE):
            print("in tt4 for")
            if R1_pre[k] < 0.5:
                R1_pre[k] = 0
            else: 
                R1_pre[k] = 1
        print("out tt4 for")
        R1_invert.append(R1_pre) # 为啥不直接 R1_invert = R1_pre
    # 我想看看 相互fuck一次 后的数据是否 令人满意
    print("R1_invert = ", R1_invert)    
    
    #print("R1_invert =",R1_invert)
    #R1_invert = [(1-a)*S_total[i]+a*R_total[i] for i in range(len(S_total))]
    R1 = R1_invert + S
    transR1 = transDNA(R1)
    R1_F = F(transR1)
    R1_fitness = get_fitness(R1_F)
   
    # 可疑之处： R1_fitness 的最小值 如果小于  fitness_aver ， 那么 while 后面的永远true
    # 也即是： while 循环永不停止
    # 迭代写法 有本质性的错误 
    while np.min(R1_fitness) < fitness_aver:
        print("enter while tt5")
        # R1_invert 将被清理，原来的数据保存在 R_total
        R_total = R1_invert.copy()
        print("tt1 R_total =", R_total)

        R1_invert.clear()
        print("tt7 R1_invert = ", R1_invert)


        print("generate new R1_invert start----")
        print("tt8 S_total == ", S_total)
        print("tt9 R_total == ", R_total)
        for i in range(len(S_total)):
            # 目的： a是比例因子，从stotal 中 和 r total 中取出一些进行交叉，交叉结果是 R1_pre
            R1_pre = [(1-a)*S_total[i][j]+a*R_total[i][j] for j in range(DNA_SIZE)]
            # 猜测： 每次 R1_pre 都一样, 故此，验证之: 确实每次都一样

            #print("R1_pre = ",R1_pre)
            # tt6 make new R1_pre for R1_invert
            for k in range(DNA_SIZE):
                if R1_pre[k] < 0.5:
                    R1_pre[k] = 0
                else: R1_pre[k] = 1

            R1_invert.append(R1_pre)
        
        print("generate new R1_invert end----:")

        print("R1_invert", R1_invert)
        #print("R1_invert's length",len(R1_invert))

        R1_new = R1_invert.copy() + S.copy()
        # tt3 add for some reason: fix that's weird tt mark3
        R1_invert = R1_new
        # tt4 add for some reason: ??
        R_total = R1_invert.copy()
        transR1_new = transDNA(R1_new)
        R1_newF = F(transR1_new)
        R1_newfitness = get_fitness(R1_newF)
        R1_fitness = R1_newfitness
        print("R1_fitness =", R1_fitness)

        
    #print("R1_F =", R1_F)
    #print("R1_newfitness =", R1_newfitness)

    return R1_new

pop = initial_origin(POP_SIZE,DNA_SIZE)  

#print("pop :", pop)
#print("pop's length :", len(pop))
#print("total =",total)
#print("total's length :", len(total))

T = transDNA(pop) # T == transedDNA
F_values = F(T) # F(T) == evaluResult == evaluateFitnessOf(transedDNA)
fitness = get_fitness(F_values) # get_fitness(F_values) == 纠正fitnessOf(evaluResult)

R1 = invert_R1(pop,fitness)
print("R1 =",R1)
#print("S_total's min:",np.argmin(get_fitness(S)))
#print("max_fitness:",float(np.max(fitness)))
#print("fitness:",fitness)
#print("F_values:",F_values)
#print("T's value :", T)
