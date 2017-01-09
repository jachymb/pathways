#!/usr/bin/env python3
import sys
from evolver import *

def literal2str(pos, neg, i):
    j = 1 << i
    if neg & pos & j:
        return "!"
    elif neg & j:
        return "-"
    elif pos & j:
        return "+"
    else:
        return "&nbsp;"


def rules2table(P, rules):
    t = '<table border="1" cellspacing="0"><tr><th></th>'
    for n in P:
        t+="<th>%s</th>" % n.name
    t+="</tr>"
    for head, rule in zip(P, rules):
        t += '<tr><th rowspan="%d">%s</th>' % (len(rule), head.name)
        for (neg, pos) in rule:
            for i in range(len(P)):
                t += "<td>%s</td>" % literal2str(pos, neg, i)
            t+="</tr><tr>"
        t = t[:-4] # Cut "<tr>"
    t += "</table>"
    return t


def rules2tableDiff(P, rules, orig):
    t = '<table border="1" cellspacing="0"><tr><th></th>'
    for n in P:
        t+="<th>%s</th>" % n.name
    t+="</tr>"
    for head, rule1, rule2 in zip(P, rules, orig):
        t += '<tr><th rowspan="%d">%s</th>' % (max_k, head.name)
        for ((neg1, pos1), (neg2, pos2)) in zip(rule1, rule2):
            for i in range(len(P)):

                l1 = literal2str(pos1, neg1, i)
                l2 = literal2str(pos2, neg2, i)
                if l1 == l2:
                    t += "<td>"
                else:
                    t += '<td style="background:red">'
                t += l1 + "</td>"
            t+="</tr><tr>"
        t = t[:-4] # Cut "<tr>"
    t += "</table>"
    return t


print_rule = False
pop_file = POP_FILE if len(sys.argv) == 1 else sys.argv[1]
if __name__ == "__main__":
    with open(POP_FILE,"rb") as dumpfile:
        pop = pickle.load(dumpfile)
        for i, ind in enumerate(pop):
            attractors_good, hamming = ind.fitness.values
            if attractors_good == TRIES:
                with open("results/%d" % i, "wb") as indf:
                    pickle.dump(ind, indf)

            with open("tables/%d" % i, "w") as tfile: 
                print(rules2tableDiff(P, ind, rules), file=tfile)
            print(ind.fitness.values)
            if print_rule:
                print ("{")
                for head, rule in zip(P, per_selur(P, ind)):
                    print("\t%s: %s" % (head.name, rule))
                print ("}")
                print()

