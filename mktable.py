from evolver import *
import os
import sys
import pickle
import csv

def to_table(P, rules, start):
    trajectory = converge(interpret_dnf, rules, start)
    n = len(P)
    return (np.tile(
                np.full(n,1) << np.arange(n),
                (len(trajectory),1)) & 
            np.tile(
                trajectory,
                (n,1)).T) > 0

def print_table(P, rules, starts, filename):
    with open(filename, "w") as f:
        #print('<style type="text/css">td { width: 10px}\nth { width: 10px}</style>')
        for start in starts:
            t = to_table(P, rules, start)
            print('<table border="1" cellpading="0" cellspacing="0"><tr><th><div style="transform:rotate(-90deg); height: 120px;width:20px;position:relative;top:50px;left:50px">Time</div></th>', file = f)
            for node in P:
                print('<th><div style="transform:rotate(-90deg); height: 120px;width:20px;position:relative;top:50px;left:50px">%s</div></th>' % node.name, file = f)
            print("</tr>", file=f)

            for i,r in enumerate(t):
                print("<tr><td>%d</td>" % (i+1), file=f)
                for c in r:
                    if c:
                        print('<td style="background:gray">&nbsp;</td>', file=f)
                    else:
                        print('<td>&nbsp</td>', file=f)
                print("</tr>", file=f)
            print("</table><br><br><br>", file=f)

if __name__ == "__main__":
    with open(sys.argv[1],"rb") as f:
        rules = pickle.load(f)


    starts = [1 << s for s in (P.DSB, P.ADD, P.ICL)]
    print(starts)
    print_table(P, rules, starts, os.path.basename(sys.argv[1])+".html")

