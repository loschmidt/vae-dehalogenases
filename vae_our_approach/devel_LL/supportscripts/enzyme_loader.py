__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/15 09:33:12"
import pandas as pd
import sys

data = pd.read_excel (sys.argv[1], sheet_name='Full Dataset')
seq_dict = {}
for i, it in data.iterrows():
    seq_dict[it['Accession']] = it['Sequence']

## Now transform sequences back to fasta
with open("xlsx_to_fasta.fasta", 'w') as file_handle:
    for seq_name, seq in seq_dict.items():
        file_handle.write(">" + "".join(seq_name) + "\n" + "".join(seq) + "\n")
print('Fasta file generate to xlsx_to_fasta.fasta')