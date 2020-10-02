__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 09:19:00"

""""
Download the multiple sequence alignment for a given Pfam ID
"""

import urllib3
import gzip
import os.path as path

from pipeline import StructChecker

class Downloader:
    def __init__(self, setuper, all=False):
        self.msa_dir = setuper.MSA_fld
        self.pfam_id = setuper.pfam_id
        self.down_seq = [setuper.rp]
        if all:
            self.down_seq = ["full", "rp75", "rp55", "rp35", "rp15", "seed"]
        if setuper.in_file == '':
            print('Using file provided in run parameter --in_file {}'.format(setuper.in_file))
            setuper.set_msa_file(setuper.in_file)
        else:
            self._download()
            setuper.set_msa_file("{0}{1}_{2}.txt".format(self.msa_dir,self.pfam_id, setuper.rp))

    def _download(self):
        for dow in self.down_seq:
            file_name = "{0}{1}_{2}.txt".format(self.msa_dir,self.pfam_id, dow)
            if path.isfile(file_name):
                print("File is already downloaded in {0}".format(file_name))
                print("If you want to download it again, please delete it from its location")
            else:
                print("Downloading the {1} multiple sequence alignment for Pfam: {0} ......".format(self.pfam_id, dow))
                http = urllib3.PoolManager()
                r = http.request('GET', 'http://pfam.xfam.org/family/{0}/alignment/{1}/gzipped'.format(self.pfam_id, dow))
                data = gzip.decompress(r.data)
                data = data.decode()
                with open(file_name, 'w') as file_handle:
                    print(data, file=file_handle)

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    downloader = Downloader(tar_dir, all=True)