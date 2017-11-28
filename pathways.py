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
from itertools import product
from tempfile import NamedTemporaryFile
import subprocess
import math, os
import random

def popcount(a):
    a -= np.bitwise_and(np.right_shift(a, 1), 0x55555555)
    a = np.bitwise_and(a, 0x33333333) + np.bitwise_and(np.right_shift(a, 2), 0x33333333)
    return np.right_shift(np.bitwise_and((np.bitwise_and(a + np.right_shift(a, 4), 0xF0F0F0F) * 0x1010101), 0xffffffff), 24)

def hamming(a, b):
    #print("callint hamming")
    return int(popcount(a^b).sum())

class Var(IntEnum):
    @classmethod
    def states(cls):
        return 1 << len(cls)

class P(Var):
    ICL = 0
    FANCM = 1
    FAcore = 2
    FANCD2I = 3
    MUS81 = 4
    FAN1 = 5
    XPF = 6
    ADD = 7
    DSB = 8
    ssDNARPA = 9
    MRN = 10
    PCNATLS = 11
    HRR = 12
    FANCD1N = 13
    RAD51 = 14
    FANCJBRCA1 = 15
    USP1 = 16
    KU = 17
    DNAPK = 18
    NHEJ = 19
    ATR = 20
    ATM = 21
    BRCA1 = 22
    p53 = 23
    CHK1 = 24
    CHK2 = 25
    H2AX = 26
    CHKREC = 27

class P2(Var):
    ICL = 0
    FAcore  = 1
    FANCD2I = 2
    NUC1 = 3
    NUC2 = 4
    ADD = 5
    DSB = 6
    TLS = 7
    FAHRR = 8
    HRR2 = 9
    NHEJ = 10
    ATR = 11
    ATM = 12
    p53 = 13
    CHKREC = 14

class X(Var):
    X0 = 0
    X1 = 1
    X2 = 2
    X3 = 3
    X4 = 4
    X5 = 5
    X6 = 6
    X7 = 7
    X8 = 8

class M(Var):
    CycD = 0
    Rb = 1
    E2F = 2
    CycE = 3
    CycA = 4
    p27 = 5
    Cdc20 = 6
    Cdh1 = 7
    UbcH10 = 8
    CycB = 9

# Force enum constants' names to namespace as logical symbols
syms = {m : Symbol(m) for s in Var.__subclasses__() for m in s.__members__}
globals().update(syms)

rules_original = {
        ICL : ICL & ~DSB,
        FANCM : ICL & ~CHKREC,
        FAcore : FANCM & (ATR | ATM) & ~CHKREC,
        FANCD2I : FAcore & ((ATM | ATR) | (H2AX & DSB)) & ~USP1,
        MUS81 : ICL,
        FANCJBRCA1 : (ICL | ssDNARPA) & (ATM | ATR),
        XPF : MUS81,
        FAN1 : MUS81 & FANCD2I,
        ADD : (ADD | (MUS81 & (FAN1 | XPF))) & ~PCNATLS,
        DSB : (DSB | FAN1 | XPF) & ~(NHEJ | HRR),
        PCNATLS : (ADD | (ADD & FAcore)) & ~USP1 ,
        MRN : DSB & ATM ,
        BRCA1 : DSB & (ATM | CHK2 | ATR) ,
        ssDNARPA : DSB & ((FANCD2I & FANCJBRCA1) | MRN) & ~RAD51,
        FANCD1N : (ssDNARPA & BRCA1) | (FANCD2I & ssDNARPA),
        RAD51 : ssDNARPA & FANCD1N,
        HRR : DSB & RAD51 & FANCD1N & BRCA1,
        USP1 : (FANCD2I | PCNATLS),
        KU : DSB & ~(FANCD2I | CHKREC),
        DNAPK : (DSB & KU),
        NHEJ : (DSB & DNAPK) | (DSB & DNAPK & KU),
        ATR : (ssDNARPA | FANCM | ATM),
        ATM : (ATR | DSB) & ~CHKREC,
        p53 : (((ATM & CHK2) | (ATR & CHK1)) | DNAPK),
        CHK1 : (ATM | ATR | DNAPK) & ~CHKREC,
        CHK2 : (ATM | ATR | DNAPK) & ~CHKREC,
        H2AX : DSB & (ATM | ATR | DNAPK) & ~CHKREC,
        CHKREC : (PCNATLS | NHEJ | HRR)  | ~DSB,
        }

rules_updated = {
        ICL: ICL & ~DSB,
        FANCM: ICL & ~CHKREC,
        FAcore: FANCM & (ATR | ATM) & ~CHKREC,
        FANCD2I: FAcore & ((ATM | ATR) | (H2AX & DSB)) & ~USP1,
        MUS81: ICL,
        FANCJBRCA1: (ICL | ssDNARPA) & (ATM | ATR),
        XPF: (MUS81 & ~FANCM) | (MUS81 & p53 & ~(FAcore & FANCD2I & FAN1)),
        FAN1: MUS81 & FANCD2I,
        ADD: (ADD | (MUS81 & (FAN1 | XPF))) & ~PCNATLS,
        DSB: (DSB | FAN1 | XPF) & ~(NHEJ | HRR),
        PCNATLS: (ADD | (ADD & FAcore)) & ~(USP1 | FAN1),
        MRN: DSB & ATM & ~((KU & FANCD2I) | RAD51 | CHKREC),
        BRCA1: DSB & (ATM | CHK2 | ATR) & ~CHKREC,
        ssDNARPA: DSB & ((FANCD2I & FANCJBRCA1) | MRN) & ~(RAD51 | KU),
        FANCD1N: (ssDNARPA & BRCA1) | (FANCD2I & ssDNARPA) & ~CHKREC,
        RAD51: ssDNARPA & FANCD1N & ~CHKREC,
        HRR: DSB & RAD51 & FANCD1N & BRCA1 & ~CHKREC,
        USP1: ((FANCD1N & FANCD2I) | PCNATLS) & ~FANCM,
        KU: DSB & ~(MRN | FANCD2I | CHKREC),
        DNAPK: (DSB & KU) & ~CHKREC,
        NHEJ: (DSB & DNAPK & XPF & ~((FANCJBRCA1 & ssDNARPA) | CHKREC)) | ((DSB & DNAPK & KU) & ~(ATM & ATR)),
        ATR: (ssDNARPA | FANCM | ATM) & ~CHKREC,
        ATM: (ATR | DSB) & ~CHKREC,
        p53: (((ATM & CHK2) | (ATR & CHK1)) | DNAPK) & ~CHKREC,
        CHK1: (ATM | ATR | DNAPK) & ~CHKREC,
        CHK2: (ATM | ATR | DNAPK) & ~CHKREC,
        H2AX: DSB & (DNAPK | ATM | ATR) & ~CHKREC,
        CHKREC: ((PCNATLS | NHEJ | HRR) & ~DSB) | (~ADD & ~ICL & ~DSB & ~CHKREC),
        }

#rules_mammalian_1 = {
#        X1: (~X3 & ~X8),
#        X2: X1,
#        X3: (X1 & ~X5 & ~(X6 & X7))|(X3 & ~X5 & ~(X6 & X7)),
#        X5: X8,
#        X6: (~X3 & ~X8) | X5 ,
#        X7: ~X6 | (X6 & X7 & (X5 | X3 | X8)),
#        X8: ~X5 & ~X6
#        }

rules_fa_2 = {
        ICL : ICL & ~DSB,
        FAcore : ICL & (ATR | ATM) & ~CHKREC,
        FANCD2I : FAcore & ((ATR | ATM) | ((ATR | ATM) & DSB)) & ~CHKREC,
        NUC1 : ICL & FANCD2I,
        NUC2 :(ICL & (ATR | ATM) & ~(FAcore & FANCD2I)) | (ICL & NUC1 & p53 & ~(FAcore & FANCD2I)),
        ADD : (NUC1 | NUC2 | (NUC1 & NUC2)) & ~TLS,
        DSB : (NUC1 | NUC2) & ~(NHEJ | FAHRR  | HRR2),
        TLS : (ADD | (ADD & FAcore)) & ~CHKREC,
        FAHRR : DSB & FANCD2I & ~(NHEJ & CHKREC),
        HRR2 : (DSB & NUC2 & NHEJ & ICL & ~(FAHRR | CHKREC)) | (DSB & NUC2 & TLS & ~(NHEJ | FAHRR | CHKREC)),
        NHEJ : (DSB & NUC2 & ~(FAHRR | HRR2 | CHKREC)),
        ATR : (ICL | ATM) & ~CHKREC,
        ATM : (ATR | DSB) & ~(CHKREC | FAcore), # AMBIGUOUS grouping in paper!!!
        p53 : ((ATR | ATM) | NHEJ) & ~CHKREC,
        CHKREC : ((TLS | NHEJ | FAHRR | HRR2) & ~DSB) | (~ADD & ~ICL & ~DSB & ~CHKREC)
        }

rules_mammalian_0 = {
        X0: (~X2 & ~X3 & ~X8)|(X4 & ~X8),
        X1: (~X0 & ~X3 & ~X8)|(X4 & ~X0 & ~X8),
        X2: X1 & ~X0,
        X3: (X1 & ~X0 & ~X5 & ~(X6 & X7))|(X3 & ~X0 & ~X5 & ~(X6 & X7)),
        X4: (~X2 & ~X3 & ~X8)|(X4 & ~(X2 & X3) & ~X8),
        X5: X8,
        X6: (~X3 & ~X8) | X5 | (X4 & ~X8),
        X7: ~X6 | (X6 & X7 & (X5 | X3 | X8)),
        X8: ~X5 & ~X6
        }

rules_mammalian_full = {
        CycD: CycD,
        Rb: (~CycD & ~CycE & ~CycA & ~CycB) | (p27 & ~CycD & ~CycB),
        E2F: (~Rb & ~CycA & ~CycB) | (p27 & ~Rb & ~CycB),
        CycE: (E2F & ~Rb),
        CycA: (E2F & ~Rb & ~Cdc20 & ~(Cdh1 & UbcH10)) | (CycA & ~Rb & ~Cdc20 & ~(Cdh1 & UbcH10)),
        p27: (~CycD & ~CycE & ~CycA & ~CycB) | (p27 & ~(CycE & CycA) & ~CycB & ~CycD),
        Cdc20: CycB,
        Cdh1: (~CycA & ~CycB) | Cdc20 | (p27 & ~CycB),
        UbcH10: ~Cdh1 | (Cdh1 & UbcH10 & (Cdc20 | CycA | CycB)),
        CycB: ~Cdc20 & ~Cdh1
        }

#paper_rules = rules_fa_2; P = P2
#paper_rules = rules_updated
#paper_rules = rules_original
paper_rules = rules_mammalian_0; P = X
#paper_rules = rules_mammalian_full; P = M

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
#assert dnf_rep(P, And(FANCD1N, ssDNARPA)).tolist() == [[0, 0b10001000000000]]
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

    l = len(explored)
    explored = []
    while state not in explored:
        explored.append(state)
        state = step(interpretation_func, state, rules)

    if canonicalize:
        s = explored.index(min(explored))
        explored = explored[s:] + explored[:s]

    #return (tuple(explored), l)
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

def model_attractors(interpretation_func, rules, subsample_size = None,
        canonicalize = True, container = set, maxsteps = None):
    np.random.seed(0)
    n_transitions = 1 << len(rules)
    if subsample_size:
        samples = np.random.randint(0, n_transitions, subsample_size, dtype=np.uint32)
    else:
        samples = np.arange(n_transitions, dtype=np.uint32)
    return container(
            complete_attractor(interpretation_func, state, rules, canonicalize = canonicalize, maxsteps = maxsteps)
            for state in samples)

def transition_model(rules) -> np.array:
    n_transitions = 1 << len(rules)
    #assert len(rules) < 32
    transitions = np.empty(n_transitions, dtype=np.uint32)
    for i in tqdm(range(n_transitions)):
        transitions[i] = step(interpret_dnf, i, rules)
        if i % 1000 == 0:
            c = Counter(transitions[:i])
            #print(np.average(list(c.values())))
            #print(np.median(list(c.values())))


    return transitions

def transition_model_as_matrix(model):
    n = len(model)
    m = np.zeros((n,n), dtype=np.uint32)
    for i,v in enumerate(model):
        m[i,v] = 1
    return m

def transition_model_to_hs(model):
    l = int(math.log2(len(model)))
    def label(i):
        return ",".join(reversed(bin(i)[2:].rjust(l, '0'))).replace("0", "False").replace("1", " True")

    s = "data :: Data %d\n" % l
    s += "data = [\n"
    s += ",\n  ".join("([%s], [%s])" % (label(a), label(b)) for a,b in enumerate(model))
    s += "\n]"
    return s

def transition_model_to_mathematica(model):
    return "Graph[{" + ",".join("DirectedEdge[%i,%i]" % p for p in enumerate(m)) + "}]"

def transition_model_to_asp(model, unknown_ratio = 0.1):
    k = len(model)
    l = int(math.log2(k))
    unknown = set(random.sample(range(k), round(k*unknown_ratio)))
    s = f"""
nvars({l}).

"""
    for state in range(k):
        p = "gt_" if state in unknown else ""
        for variable in range(l):
            i = min(1, model[state] & (1 << variable))
            s += f"{p}observation({state}, {variable}, {i}).\n"
        s += f"{p}transition({state}, {model[state]}).\n"
    return s

def transition_model_to_dot(model):
    l = int(math.log2(len(model)))
    w = 1
    while w*w < len(model):
        w*=2

    def label(i):
        return bin(i)[2:].rjust(l, '0')
    def pos(i):
        return "%d,%d!" % divmod(i,w)

    g = "strict digraph {\n  "
    #g += ";\n  ".join(f'q{f} [label="{label(f)}", pos="{pos(f)}"]' for f in range(len(model)))
    g += ";\n  ".join(f'q{f} [label="{label(f)}"]' for f in range(len(model)))
    g += ";\n  "
    g += ";\n  ".join(f'q{f} -> q{t}' for f,t in enumerate(model))
    g += "\n}"
    return g

def blif2rules(blifData):
    v = None
    rules = []
    last = 0
    for line in blifData:
        line = line[:line.find('#')].strip()
        if line:
            words = line.split(' ')
            if words[0] == '.v':
                v = int(words[1])
            elif words[0] == '.n':
                assert last + 1 == int(words[1])
                last += 1
                dependent = [int(x)-1 for x in words[3:]]
                rules.append((dependent, []))
            else:
                assert words[1] == '1'
                rules[-1][1].append(words[0])
    assert v == len(rules)
    assert v < 32
    maxterms = max(len(r[1]) for r in rules)
    rules_np = np.zeros((v, maxterms, 2), dtype=np.uint32)
    for i, (dependent, dnf) in enumerate(rules):
        for j, conj in enumerate(dnf):
            for v,b in zip(dependent, conj):
                if b == '0':
                    rules_np[i][j][0] |= 1 << v
                elif b == '1':
                    rules_np[i][j][1] |= 1 << v
    return rules_np







interpret_logicalP = partial(interpret_logical,P)

rules_sym = {head : to_dnf(body, simplify=True) for head,body in paper_rules.items()}
rules_logical = [r[1] for r in sorted(paper_rules.items(), key=lambda x: int(P[str(x[0])]))]
rules_logical2 = [to_dnf(r) for r in rules_logical]
rules = rules_rep(P, rules_sym, same_k = True)
max_k = max(dnf_len(dnf) for dnf in rules_sym.values())

def all_clauses(n_vars):
    return (
            (x,y)
            for (x,y)
            in product(np.arange(2**n_vars, dtype=np.uint32), repeat=2)
            if x & y == Z and x | y)

def all_dnfs(n_vars, k):
    return map(np.array, product(all_clauses(n_vars), repeat=k))

def all_rules(n_vars, k):
    return product(all_dnfs(n_vars, k), repeat=n_vars)

def has_single_fixpoint(rules):
    def cont(i):
        s = next(i)
        if len(s) != 1:
            return False
        for z in i:
            if z != s:
                return False
        return True

    return model_attractors(interpret_dnf, rules, canonicalize=False, container=cont)

def is_contraction(rules):
    ub = 2**len(rules)
    for i in map(np.uint32, range(ub)):
        st1 = step(interpret_dnf, i, rules)
        for j in map(np.uint32, range(ub)):
            if i != j:
                st2 = step(interpret_dnf, j, rules)
                ham_a = popcount(i ^ j)
                ham_b = popcount(st1 ^ st2)
                if ham_b >= ham_a: return False
    return True


def main():
    for i,attractor in enumerate(model_attractors(interpret_logicalP, rules_logical2, 200)):
        print(i+1)
        for a in attractor:
            print(("{0:0"+str(len(P))+"b}").format(a))

    print("-" * 20)

    for i,attractor in enumerate(model_attractors(interpret_dnf, rules, 200)):
        print(i+1)
        for a in attractor:
            print(("{0:0"+str(len(P))+"b}").format(a))

def plot(k, c):
    print("-"*30)
    print(k, c)
    isos = set()
    cnt = 0
    for i, rules in enumerate(all_rules(k, c)):
        m = tuple(transition_model(rules))
        if m in isos:
            continue
        else:
            isos.add(m)
        if not has_single_fixpoint(rules): continue

        dot = transition_model_to_dot(m)
        if not os.path.isdir(f"dot/{k}/{c}"): os.makedirs(f"dot/{k}/{c}")
        if not os.path.isdir(f"png/{k}/{c}"): os.makedirs(f"png/{k}/{c}")
        md = f"dot/{k}/{c}/{i}.dot"
        mn = f"png/{k}/{c}/{i}.png"
        with open(md, "w") as f:
            f.write(dot)
            #f.close()
        subprocess.check_call(["dot", "-Kfdp", "-Tpng", md, "-o", mn])
        cnt += 1
        print(i, "%.2f" % (cnt / len(isos)))
        #subprocess.check_call(["display", mn])

#plot(2,1)
#plot(2,2)
#plot(2,3)
#plot(3,1)
#plot(3,2)
#plot(3,3)
m = transition_model(rules)
print(transition_model_to_asp(m, 0.2))
