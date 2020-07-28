
from .pipeline import StructChecker

class MSA:
    def __init__(self, setuper : StructChecker):
        self.msa_file = setuper.rp_dir
        self.values = {"ref" : setuper.ref_seq, "ref_n" : setuper.ref_n, }
        self._load_msa()

    def _load_msa(self):
        seq_dict = {}
        with open(self.msa_file, 'r') as file_handle:
            for line in file_handle:
                if line[0] == "#" or line[0] == "/" or line[0] == "":
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                seq_id, seq = line.split()
                seq_dict[seq_id] = seq.upper()
        self.seq_dict = seq_dict

    def _ref_filtering(self):
        query_seq = self.seq_dict[self.values["ref_n"]]  ## with gaps
        idx = [s == "-" or s == "." for s in query_seq]
        for k in self.seq_dict.keys():
            self.seq_dict[k] = [self.seq_dict[k][i] for i in range(len(self.seq_dict[k])) if idx[i] == False]
        query_seq = self.seq_dict[self.values["ref_n"]]  ## without gaps

        ## remove sequences with too many gaps
        len_query_seq = len(query_seq)
        seq_id = list(self.seq_dict.keys())
        num_gaps = []
        for k in seq_id:
            num_gaps.append(self.seq_dict[k].count("-") + self.seq_dict[k].count("."))
            if self.seq_dict[k].count("-") + self.seq_dict[k].count(".") > 0.2 * len_query_seq:
                self.seq_dict.pop(k)

    def get_best_ref_seq(self):

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    msa = MSA(tar_dir)