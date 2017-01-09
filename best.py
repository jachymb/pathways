#!/usr/bin/env python3
import sys
from evolver import *

print_rule = False
pop_file = POP_FILE if len(sys.argv) == 1 else sys.argv[1]
if __name__ == "__main__":
    with open(POP_FILE,"rb") as dumpfile:
        pop = pickle.load(dumpfile)
        for ind in pop:
            print(ind.fitness.values)
            if print_rule:
                print ("{")
                for head, rule in zip(P, per_selur(P, ind)):
                    print("\t%s: %s" % (head.name, rule))
                print ("}")
                print()

