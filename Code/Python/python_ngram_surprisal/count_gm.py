
"""
to run this file:

python3 count_gm.py --gm [number of ngram] --file [corpora] --outp [output path]

output will be all ngram strings in the corpora and their occurrance

"""


import math as m
from argparse import ArgumentParser


p = ArgumentParser()
p.add_argument("--gm", type = int)
p.add_argument("--file")
p.add_argument("--outp")
args = p.parse_args()



ix = {}
f1 = open(args.file, 'r')
for l in f1:
	l = l
	for i in range(args.gm-1):
		l = "<bos> " + l
	l = l.lower().split()	
	for i in range(len(l)+1-args.gm):
		tmp_str = l[i]
		for c in range(1, args.gm):
			tmp_str = tmp_str + " " +l[i+c]
		if tmp_str not in ix:
			ix[tmp_str] = 0
		ix[tmp_str] += 1


f1.close()
fo = open(args.outp, 'w', encoding = 'utf-8')
for i in ix:
	fo.write(i + " " + str(ix[i]) + "\n")

fo.close()




