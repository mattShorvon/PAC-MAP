from spn.utils.graph import full_binarization
from spn.io.file import from_file, to_file
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--spn_file")
args = parser.parse_args()
SPN_FILE = args.spn_file
SPN_FILE2 = SPN_FILE[:-4]+".spn2"
SPN_FILE3 = SPN_FILE[:-4]+".spn2l"
spn = from_file(SPN_FILE)
spn = full_binarization(spn)
spn.fix_scope()
spn.fix_topological_order()
print("Reading "+SPN_FILE)
to_file(spn, SPN_FILE2)
fin = open(SPN_FILE2, "rt")
fout = open(SPN_FILE3, "wt")
for line in fin:
	fout.write(line.replace('indicator', 'l'))
fin.close()
fout.close()
print("Writing "+SPN_FILE3)