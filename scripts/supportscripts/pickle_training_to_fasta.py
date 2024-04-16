import pickle
import sys

with open(sys.argv[1], "rb") as f:
    msa = pickle.load(f)
    
with open("converted.fasta", 'w') as file_handle:
    for key, sequence in msa.items():
        file_handle.write(">" + key + "\n")
        n = 80
        seq = "".join(sequence)
        for i in range(0, len(seq), n):
            file_handle.write(seq[i:i+n] + "\n")
