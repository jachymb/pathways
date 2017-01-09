#!/usr/bin/env python3
##!/bin/sh
#"exec" "python3" "-m" "scoop" "$0"

from collections import Counter
from datetime import datetime
from deap import creator, base, tools, algorithms
from deap.tools.emo import assignCrowdingDist
from itertools import count
from numpy.random import randint, random, seed
from pathways import *
from scoop import futures
import multiprocessing
import numpy as np
import pickle
import shutil, os, sys
import sys

n = len(P)
all_present = (1 << n) - 1
rules_shape = n, max_k, 2

MU = 1024 # Needs to be divisible by 4!
CXPB = 0.9

TRIES = 8

POP_FILE = "pop.dat"
POP_FILE = POP_FILE if len(sys.argv) == 1 else sys.argv[1]

def uniformly_random_individual():
    return randint(all_present, dtype=np.uint32, size=rules_shape)

def random_individual(k=4):
    data = np.full((n, max_k, 2), (1 << n) - 1, dtype=np.uint32)
    for _ in range(k):
        data &= uniformly_random_individual()
    return data

def mutate_k(ind, max_mutations = max_k, indpb = 0.9):
    #print("calling mutate_k")
    for _ in range(max_mutations):
        if random() < indpb:
            ind[randint(n)][randint(max_k)][randint(2)] ^= I << np.uint32(randint(n)) # Flip bit
    return ind

def mutate_anywhere(ind, indpb = 0.02):
    #print("calling mutate_anywhere")

    for m in range(n):
        for k in range(max_k):
            for i in (0,1):
                for j in range(n):
                    if random() < indpb:
                        ind[m][k][i] ^= I << np.uint32(j) # Flip bit
    return ind

def popcount(a):
    a -= np.bitwise_and(np.right_shift(a, 1), 0x55555555)
    a = np.bitwise_and(a, 0x33333333) + np.bitwise_and(np.right_shift(a, 2), 0x33333333)
    return np.right_shift(np.bitwise_and((np.bitwise_and(a + np.right_shift(a, 4), 0xF0F0F0F) * 0x1010101), 0xffffffff), 24)

def hamming(a, b):
    #print("callint hamming")
    return int(popcount(a^b).sum())

def attractor_appearance(rules : np.ndarray, attractor : tuple, maxsteps = 16):
    #print("callint attractor_appearance")
    return model_attractors(interpret_dnf, rules, TRIES,
            canonicalize = True,
            container = Counter,
            maxsteps = maxsteps)[attractor]

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    #size = len(ind1) = n
    cxpoint1 = randint(1, n)
    cxpoint2 = randint(1, n - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2

def cxUniformCopy(ind1, ind2, indpb=0.5):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(n):
        if random() < indpb:
            ind1[i], ind2[i] = ind2[i].copy(), ind1[i].copy()

    return ind1, ind2

def cxOnePointCopy(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    cxpoint = randint(1, n - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:].copy(), ind1[cxpoint:].copy()

    return ind1, ind2

DESIRED_ATTRACTOR = (0b0000000000000000000000000000, 0b1000000000000000000000000000)

RULES_SYM = {head : to_dnf(body, simplify=True) for head,body in rules_original.items()}
RULES = rules_rep(P, RULES_SYM, same_k = True)
assert RULES.shape == rules.shape

def evaluate(ind):
    #print("calling evaluate")
    return attractor_appearance(ind, DESIRED_ATTRACTOR), hamming(ind, RULES)

creator.create("Fitness", base.Fitness, weights=(1, -1))
creator.create("Individual", np.ndarray, fitness=creator.Fitness)
creator.Individual.__hash__ = lambda self: int(self.sum()) # Monkey patch hack for other stuff to work

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, random_individual)

#toolbox.register("map", futures.map)
processes = int(os.getenv("PROCESSES", 2))
print("Running on %d processes" % processes)
pool = multiprocessing.Pool(processes)
toolbox.register("map", pool.map)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
#toolbox.register("mate", cxUniformCopy)
#toolbox.register("mate", cxOnePointCopy)
toolbox.register("mutate", mutate_k)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selNSGA2)
def spea2(pop, mu):
    assignCrowdingDist(pop)
    return tools.selSPEA2(pop, mu)
#toolbox.register("select", spea2)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

def main(seed_=0):
    #seed(seed_)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    if os.path.isfile(POP_FILE):
        with open(POP_FILE,"rb") as dumpfile:
            pop = pickle.load(dumpfile)
    else:
        pop = toolbox.population(n=MU)

    #pop[0] = creator.Individual(rules)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    try:
        for gen in count(1):

            #with open("gen-%d.pop" % gen,"wb") as dumpfile:
            if os.path.isfile(POP_FILE):
                shutil.move(POP_FILE, POP_FILE+".bak")

            with open(POP_FILE,"wb") as dumpfile:
                pickle.dump(pop, dumpfile)
            # Vary the population
        
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random() <= CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(str(datetime.now()))
            print(logbook.stream)
            #print([ind.fitness.values for ind in pop])
    except KeyboardInterrupt:
        return pop, logbook
    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook

if __name__ == "__main__":
    pop, stats = main()
    for ind in pop:
        print(ind.fitness.values)

    print(stats)

