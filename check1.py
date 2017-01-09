#!/usr/bin/env python3
import sys
from evolver import *

if __name__ == "__main__":
    fname = sys.argv[1]
    with open(fname, "rb") as indfile:
        ind = pickle.load(indfile)

    attractors = model_attractors_exhaustive(P, interpret_dnf, ind)
    with open(fname + "-attractors", "w") as f:
        print(attractors, file=f)
