__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 09:19:00"

""""
Download the multiple sequence alignment for a given Pfam ID
"""

import urllib3
import gzip

from pipeline import StructChecker

class Downloader:
    def __init__(self, setuper, all=False):
        self.msa_dir = setuper.MSA_fld
        self.pfam_id = setuper.pfam_id
        self.down_seq = [setuper.rp]
        if all:
            self.down_seq = ["full", "rp75", "rp55", "rp35", "rp15", "seed"]
        self._download()

    def _download(self):
        for dow in self.down_seq:
            print("Downloading the {1} multiple sequence alignment for Pfam: {0} ......".format(self.pfam_id, dow))
            http = urllib3.PoolManager()
            r = http.request('GET', 'http://pfam.xfam.org/family/{0}/alignment/{1}/gzipped'.format(self.pfam_id, dow))
            data = gzip.decompress(r.data)
            data = data.decode()
            with open("{0}{1}_{2}.txt".format(self.msa_dir,self.pfam_id, dow), 'w') as file_handle:
                print(data, file=file_handle)

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    downloader = Downloader(tar_dir, all=True)