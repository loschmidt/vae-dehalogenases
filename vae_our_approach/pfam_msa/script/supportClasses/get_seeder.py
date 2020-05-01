__author__ = "Pavel Kohout  <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/04/12 23:27:00"

import argparse

parser = argparse.ArgumentParser(description = "Process given MSA")
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")

args = parser.parse_args()
pfam_id = args.Pfam_id


file_name = "./MSA/{0}_seed.txt".format(pfam_id)

## read all the sequences into a dictionary
seq_dict = {}
with open(file_name, 'r') as file_handle:
    for line in file_handle:
        if line[0] == "#" or line[0] == "/" or line[0] == "":
            continue
        line = line.strip()
        if len(line) == 0:
            continue
        seq_id, seq = line.split()
        seq_dict[seq_id] = seq.upper()


max_len = 0
max_name = "Not found"

## Chose the longest except gaps
for name, seq in seq_dict.items():
    seq = [seq[i] for i in range(len(seq)) if seq[i] != "-" and seq[i] != "."]
    if len(seq) > max_len:
        max_len = len(seq)
        max_name = name

print("="*60)
print("Founded seed is {0} with length {1}".format(max_name, max_len))
print("="*60)