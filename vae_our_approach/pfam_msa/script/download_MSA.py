__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/08 18:50:39"

"""
Download the multiple sequence alignment for a given Pfam ID
"""

import urllib3
import gzip
import sys
import argparse
from Bio import SeqIO

FASTA_GENERATE = False

parser = argparse.ArgumentParser(description = "Download the full multiple sequence alignment (MSA) in Stockholm format for a Pfam_id.")
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
args = parser.parse_args()
pfam_id = args.Pfam_id

down_seq = ["full", "rp75", "rp55", "rp35", "rp15", "seed"]

for dow in down_seq:
    print("Downloading the {1} multiple sequence alignment for Pfam: {0} ......".format(pfam_id, dow))
    http = urllib3.PoolManager()
    r = http.request('GET', 'http://pfam.xfam.org/family/{0}/alignment/{1}/gzipped'.format(pfam_id, dow))
    data = gzip.decompress(r.data)
    data = data.decode()
    with open("./MSA/{0}_{1}.txt".format(pfam_id, dow), 'w') as file_handle:
        print(data, file = file_handle)

if FASTA_GENERATE:
    stockholm_file_name = "./MSA/{0}_full.txt".format(pfam_id)
    fasta_file_name = "./MSA/{0}_full.fasta".format(pfam_id)

    records = SeqIO.parse(stockholm_file_name, "stockholm")
    count = SeqIO.write(records, fasta_file_name, "fasta")
    print("Converted %i records" % count)

    print("Convertion done")