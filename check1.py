#!/usr/bin/env python3
import sys, os
from evolver import *
if os.getenv("USE_TQDM", False):
    from tqdm import tqdm
else:
    tqdm = lambda x, **kw: x

if __name__ == "__main__":
    fname = sys.argv[1]
    with open(fname, "rb") as indfile:
        ind = pickle.load(indfile)

    if all(tqdm(
            hasSingleAttractor(P, interpret_dnf, ind, DESIRED_ATTRACTOR),
            total = P.states())):
        print("Yea!")
    else:
        print("Nay!")
