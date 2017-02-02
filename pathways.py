#!/usr/bin/python3
from enum import IntEnum
import numpy as np
from tqdm import tqdm
from functools import lru_cache, partial
from sympy.core import symbols
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Or, And, Not, to_dnf, to_cnf, Boolean
from sympy.core.function import FunctionClass
from collections import Counter
from data import *

Z = np.uint32(0)
A = ~Z
I = np.uint32(1)

def dnf_len(dnf : Boolean):
    if type(dnf) is Or: return len(dnf.args)
    else: return 1

def dnf_rep(p : IntEnum, dnf : Boolean, k : int = None) -> np.array:
    if k is None:
        k = dnf_len(dnf)
    #assert len(p) < 32
    rep = np.zeros((k,2), dtype=np.uint32)

    def add_literal(i : int, literal : Boolean):
        if type(literal) is Symbol:
            rep[i, 1] |= 1 << p[str(literal)]
        elif type(literal) is Not:
            var, = literal.args
            rep[i, 0] |= 1 << p[str(var)]
        else:
            raise TypeError(repr(literal) +" is not a valid literal")

    def add_conj(i : int, conj : FunctionClass):
        for sym in conj.args:
            add_literal(i, sym)

    if type(dnf) is And:
        add_conj(0, dnf)
    elif type(dnf) is Or:
        for i,conj in enumerate(dnf.args):
            if type(conj) is And:
                add_conj(i, conj)
            else:
                add_literal(i, conj)
    else:
        add_literal(0, dnf)
    return rep
# Test dnf_rep
assert dnf_rep(P, And(FANCD1N, ssDNARPA)).tolist() == [[0, 0b10001000000000]]
#print(dnf_rep(P, And(ATR, FAcore, Not(USP1))))
#print(np.array([[1 << 16, (1 << 2) | (1 << 20)]], dtype=np.uint32))

def per_fnd(q : IntEnum, rep : np.array):
    conjs = []
    for c in rep:
        conj = []
        n, p = c
        if p == n == Z: continue
        for i in range(0, len(q)):
            j = I << i
            if p & j:
                conj.append(Symbol(q(i).name))
            if n & j:
                conj.append(Not(Symbol(q(i).name)))
        conjs.append(And(*conj))
    return Or(*conjs)

# Super optimized
def interpret_conj(i : np.uint32, c : np.ndarray) -> bool:
    #print("interpret_conj(%s, %s)" % (repr(i),repr(c)))
    n, p = c
    # We need empty conjunction to evaluate to False because it should not be there in the first place
    return ((n == (~i & n)) and (p == (i & p))) and not (n == p == Z)
    #j = 1
    #for l in c:
    #    if l and ((not (i & j)) == (l >= 0)):
    #        return False
    #    j <<= 1
    #return True

# Tests
#assert interpret_conj(0b111, [1,1,0])
#assert interpret_conj(0b101, [1,0,1])
#assert interpret_conj(0b101, [1,-1,1])
#assert not interpret_conj(0b101, [1,1,0])
#assert not interpret_conj(0b101, [-1,-1,0])
#assert not interpret_conj(0b000, [-1,1,0])
#assert not interpret_conj(0b00, [1, 0])
#assert not interpret_conj(0b00, [0, 1])
assert interpret_conj(0b111, [0b000, 0b110])
assert interpret_conj(0b101, [0b000, 0b101])
assert interpret_conj(0b101, [0b010, 0b101])
assert not interpret_conj(0b101, [0b000,0b110])
assert not interpret_conj(0b101, [0b110,0b000])
assert not interpret_conj(0b000, [0b100,0b010])
assert not interpret_conj(0b00, [0b00,0b10])
assert not interpret_conj(0b00, [0b00,0b01])

def interpret_logical(p : IntEnum, i: int, formula : Boolean) -> bool:
    ftype = type(formula)
    if ftype is And:
        return all(interpret_logical(p, i, a) for a in formula.args)
    elif ftype is Or:
        return any(interpret_logical(p, i, a) for a in formula.args)
    elif ftype is Not:
        return not interpret_logical(p, i, *formula.args)
    elif ftype is Symbol:
        return bool(i & (1 << p[str(formula)]))

def interpret_dnf(i: int, dnf : np.ndarray) -> bool:
    #assert dnf.ndim == 2
    #assert dnf.shape[1] == 2
    #assert not np.any(np.all(dnf == 0, axis=1)) # No empty terms

    for c in dnf:
        if interpret_conj(i, c):
            return True
    return False

def interpret_dnf_unint(i: np.uint32 , dnf : np.ndarray) -> np.uint32:
    for c in dnf:
        if interpret_conj(i, c):
            return I
    return Z


# Test
assert interpret_dnf(0b00, np.array([[0b10,0b00],[0b00,0b01]], dtype=np.int8))
assert not interpret_dnf(0b00, np.array([[0b00,0b10],[0b00,0b01]], dtype=np.int8))
#assert interpret_dnf(0b11, np.array([
#    [0b10000000000000000,0b000001000000000000000000100],
#    [0b10000000000000000,0b000000100000000000000000100],
#    [0b10000000000000000,0b100000000000000000100000100]],
#    dtype=np.uint32)) == True

def rules_rep(p : IntEnum, rules : dict, same_k = False):
    keys = sorted(rules.keys(), key=lambda sym: int(p[str(sym)]))
    if same_k:
        max_k = max(dnf_len(dnf) for dnf in rules.values())
        reps = [dnf_rep(p, rules[key], k = max_k) for key in keys]
        return np.array(reps)
    else:
        reps = (dnf_rep(p, rules[key]) for key in keys)
        return tuple(reps)

def per_selur(p : IntEnum, rules : np.ndarray):
    return list(map(partial(per_fnd, p), rules))

#step_cache = np.full(2**len(P), A, dtype=np.uint32)
def step(interpretation_func, state : np.uint32, rules) -> np.uint32:
    #assert isinstance(rules, np.ndarray)
    r = Z
    i = Z
    for rule in rules:
        #assert isinstance(rule, np.ndarray), rule
        #assert rule.ndim == 2, (rule.shape, i, rules.shape)
        r |= (interpretation_func(state, rule) << i)
        i += 1
    return r

def step_dnf(state : np.uint32, rules) -> np.uint32:
    r = Z
    i = Z
    for rule in rules:
        r |= (interpret_dnf(state, rule) << i)
        i += 1
    return r


def attractor(interpretation_func, state : np.uint32, rules) -> int:
    #assert isinstance(rules, np.ndarray)
    explored = set()
    while state not in explored:
        explored.add(state)
        state = step(interpretation_func, state, rules)
    return state

def reachable(interpretation_func, state : np.uint32, state_to : np.uint32, rules) -> bool:
    explored = set()
    while state not in explored:
        if state == state_to:
            return True
        explored.add(state)
        state = step(interpretation_func, state, rules)
    return state == state_to

def complete_attractor(interpretation_func, state : np.uint32, rules, canonicalize = False, maxsteps = None) -> tuple:
    explored = set()
    if maxsteps is None:
        while state not in explored:
            explored.add(state)
            state = step(interpretation_func, state, rules)
    else:
        while maxsteps > 0:
            explored.add(state)
            state = step(interpretation_func, state, rules)
            if state in explored:
                break
            maxsteps -= 1
        else:
            return None

    explored = []
    while state not in explored:
        explored.append(state)
        state = step(interpretation_func, state, rules)

    if canonicalize:
        s = explored.index(min(explored))
        explored = explored[s:] + explored[:s]

    return tuple(explored)

def run_network(interpretation_func, state : np.uint32, rules):
    yield state
    while True:
        state = step(interpretation_func, state, rules)
        yield state

def model_attractors_exhaustive(P, interpretation_func, rules, 
        canonicalize = True):
    for state in range(P.states()):
        yield complete_attractor(interpretation_func, state, rules, canonicalize = canonicalize, maxsteps = None)

# use with "all"
def hasSingleAttractor(P, rules, desired_attractor):
    m = np.full(P.states(), False, dtype=bool)
    for f, t in zip(desired_attractor, (desired_attractor+(desired_attractor[-1],))[1:]):
        if step_dnf(f, rules) != t:
            #return False
            raise ValueError("Invalid attractor")
        else:
            m[f] = True

    visited = set()
    for current in range(P.states()):
        visited.clear()
        while current not in visited:
            if m[current]:
                for v in visited: m[v] = True
                yield True
                break
            visited.add(current)
            current = step_dnf(current, rules)
        else:
            yield False

def model_attractors(interpretation_func, rules, subsample_size : int,
        canonicalize = True, container = set, maxsteps = None):
    np.random.seed(0)
    n_transitions = 1 << len(rules)
    samples = np.random.randint(0, n_transitions, subsample_size, dtype=np.uint32)
    return container(
            complete_attractor(interpretation_func, state, rules, canonicalize = canonicalize, maxsteps = maxsteps)
            for state in samples)

def converge(interpretation_func, rules, state):
    explored = []
    while state not in explored:
        explored.append(state)
        state = step(interpretation_func, state, rules)

    return explored


def transition_model(rules) -> np.array:
    n_transitions = 1 << len(rules)
    #assert len(rules) < 32
    transitions = np.empty(n_transitions, -1, dtype=np.uint32)
    for i in range(n_transitions):
        transitions[i] = step(i, rules)
    return transitions

interpret_logicalP = partial(interpret_logical,P)

rules_sym = {head : to_dnf(body, simplify=True) for head,body in paper_rules.items()}
rules_logical = [r[1] for r in sorted(paper_rules.items(), key=lambda x: int(P[str(x[0])]))]
rules_logical2 = [to_dnf(r) for r in rules_logical]
rules = rules_rep(P, rules_sym, same_k = True)
max_k = max(dnf_len(dnf) for dnf in rules_sym.values())

# test
#assert per_selur(P,rules_rep(P,rules_logical2)) == rules_logical2

if __name__ == "__main__":

    for i,attractor in enumerate(model_attractors(interpret_logicalP, rules_logical2, 200)):
        print(i+1)
        for a in attractor:
            print(("{0:0"+str(len(P))+"b}").format(a))

    print("-" * 20)

    for i,attractor in enumerate(model_attractors(interpret_dnf, rules, 200)):
        print(i+1)
        for a in attractor:
            print(("{0:0"+str(len(P))+"b}").format(a))


