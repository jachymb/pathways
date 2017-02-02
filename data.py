from enum import IntEnum
from sympy.core.symbol import Symbol

class ProteinList(IntEnum):
    @classmethod
    def states(cls):
        return 1 << len(cls)

class P(ProteinList):
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

class P2(ProteinList):
    ICL = 0
    FAcore = 1
    FANCD2I = 2
    NUC1 = 3
    NUC2 = 4
    ADD = 5
    DSB = 6
    ATR = 7
    ATM = 8
    p53 = 9
    TLS = 10
    FAHRR = 11
    HRR2 = 12
    NHEJ = 13
    CHKREC = 14

# Force enum constants' names to namespace as logical symbols
syms = {m : Symbol(m) for m in P.__members__}
globals().update(syms)
FANCJMLH1 = FANCJBRCA1 

# Et voila,
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

#paper_rules = rules_updated
paper_rules = rules_original

changes = {
        P2.ICL: {ICL},
        P2.FAcore : {FANCM, FAcore},
        P2.FANCD2I : {FANCD2I},
        P2.NUC1: {MUS81},
        P2.NUC2: {XPF, FAN1},
        P2.ADD: {ADD},
        P2.DSB: {DSB},
        P2.ATR: {ATR, CHK1, H2AX},
        P2.ATM: {ATM,CHK2, H2AX},
        P2.p53: {p53},
        P2.TLS: {PCNATLS},
        P2.FAHRR: {FANCJMLH1, MRN, BRCA1, FANCD1N, RAD51, HRR, ssDNARPA},
        P2.HRR2: set(),
        P2.NHEJ: {KU, DNAPK, NHEJ},
        P2.CHKREC: {CHKREC, USP1}
        }

def transform_rules(P1, P2, rules, changes):
    new_rules = {}
    for key, values in change.items():
        pass

         
