import sys

our = ''

with open(sys.argv[1], "r") as file_handle:
    for line in file_handle:
        if line[0] == "#" or line[0] == "/" or line[0] == "":
            continue
        line = line.strip()
        if len(line) == 0:
            continue
        seq_id, seq = line.split()
        if seq_id == 'OQO00754.1':
            our += seq

# with open(sys.argv[2], 'rw') as file_handle:
#     for line in file_handle:
#         if line == 'OQO00754.1':
#             ## remove
#     file_handle.write(">" +  + "\n" + "".join(seq) + "\n")

print('>OQO00754.1 \n{}'.format(our))