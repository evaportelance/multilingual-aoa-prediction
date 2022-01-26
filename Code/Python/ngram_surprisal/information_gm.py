
"""
to run this file:

python3 information_gm.py --gm [number of ngram] --file [output of count_gm.py] --outp [output path]

output will be all ngram surprisal for all words in the corpora

"""
from argparse import ArgumentParser
import re
import math as m
#value = re.compile(r'^[A-Za-z][a-z]+$')


p = ArgumentParser()
p.add_argument("--gm", type = int)
p.add_argument("--file")
p.add_argument("--outp")

args = p.parse_args()



ix = {} # ix[w] = {cnt:0, ix_bg:[], info:0, info_g:[], N:0}
total = 0
bg = {} # b g['bg'] = 0
cnt = 'cnt'
ix_bg = 'ix_bg'
freq = 'freq'
info = 'info'
info_g = 'info_g'
N = 'N'

file = open(args.file,'rb')
for line in file:
	line = line.decode('utf-8').split()
	if len(line) < args.gm+1:
		continue
	for w in line[:-1]:
		w = w.lower()
	tmp_bg = " ".join(line[:args.gm-1])
	tmp_tg = line[args.gm-1]
	tmp_cnt = int(line[args.gm])
	if tmp_tg not in ix:
		ix[tmp_tg] = {cnt:0, ix_bg:{}, freq:0, info:0, info_g:[], N:0}
	ix[tmp_tg][cnt] += int(tmp_cnt)
	ix[tmp_tg][N] += tmp_cnt
	ix[tmp_tg][ix_bg][tmp_bg] = tmp_cnt
	if tmp_bg not in bg:
		bg[tmp_bg] = int(tmp_cnt)
	else:
		bg[tmp_bg] += int(tmp_cnt)
	total += tmp_cnt
file.close()

fileout1 = open(args.outp, 'w', encoding = 'utf-8')
fileout1.write("word,info_float,len,cnt,ngm\n")
info_total = 0
for i in ix:
	b = ix[i][ix_bg]
	for tmp_b in b:
		ix[i][info] += (-m.log(b[tmp_b] / bg[tmp_b], 2) * b[tmp_b])
	info_total += ix[i][info]
	ix[i][info] = ix[i][info] / ix[i][N]
	ix[i][freq] = -m.log(ix[i][cnt] / total, 2)
	fileout1.write(i + ',' + str(ix[i][info]) + ',' + str(len(i)) + ',' + str(ix[i][cnt]) + "," + str(args.gm) + '\n')
#fileout1.write(str(info_total) + "\n")	
fileout1.close()


