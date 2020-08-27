__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/10 11:30:00"

from download_MSA import Downloader
from msa_prepar import MSA
from pipeline import StructChecker
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains

from time import sleep
from PIL import Image
from io import BytesIO

import re

## Alignment
from Bio import SeqIO
from Bio.Align.Applications import MafftCommandline
import tempfile

class PageHandler:
    def __init__(self, ec=None):
        self.scr_n = 0
        if ec is None:
            ## Uniprot
            self.URL = "https://www.uniprot.org"
            self.msg = "Uniprot"
            self.driver = self._init_driver()
            self.table = TableLoader(self.driver)
        else:
            ## EnzymeMiner
            self.ec = ec
            self.URL = "https://loschmidt.chemi.muni.cz/enzymeminer/ec/" + self.ec
            self.msg = "Swissprot"
            self.driver = self._init_driver()
            self.table = TableLoader(self.driver)

    def translate(self, uniCode):
        request = self.URL + "/uniprot/" + uniCode
        self.driver.get(request)
        sleep(0.3) # Wait for loading
        name = self.driver.find_element_by_xpath("//section[@id='page-header']//h2//span").text[1:-1] ## Without brackets
        return name

    def screen(self):
        self.driver.fullscreen_window()
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight-100);")
        shot = self.driver.get_screenshot_as_png()
        image = Image.open(BytesIO(shot))
        name = "screenshot{}.png".format(self.scr_n)
        image.save(name, 'PNG', optimize=True, quality=95)
        self.scr_n += 1

    def select_all(self):
        ## Get buttons and select desired one
        buttons = self.driver.find_elements_by_class_name("btn-secondary")
        ## Find button select all
        for btn in buttons:
            if btn.text == 'Select all':
                btn.click()
                print("Pressing select all button... ")
                sleep(1)
                break

    def switch_tabs(self):
        """Switching tab to custom sequences in EnzymeMiner"""
        tabs = self.driver.find_elements_by_xpath("//li[@class='nav-item']")
        self.driver.execute_script("arguments[0].scrollIntoView();", tabs[1]);
        mouse_hover = ActionChains(self.driver).move_to_element(tabs[1])
        mouse_hover.click()
        mouse_hover.perform()
        print("Switching to custom sequences tab...")
        sleep(1)

    def get_queries(self):
        return self.table.select_queries()

    def get_catalytic_res(self):
        return self.table.get_catalytic_res_names()

    def close(self):
        self.driver.close()

    def _init_driver(self):
        opts = Options()
        opts.headless = True
        assert opts.headless  # Operating in headless mode
        browser = Firefox(options=opts)
        browser.get(self.URL)
        print("Loading {} tab page...".format(self.msg))
        sleep(0.5)  ## Just wait to load the page, stupid
        return browser

    def get_uniprot_seq(self, uniCode):
        request = self.URL + "/uniprot/" + uniCode + '.fasta'
        self.driver.get(request)
        sleep(0.3)  # Wait for loading
        sequence = ''.join(self.driver.find_element_by_xpath("//pre").text.split('\n')[1:]) ## Remove first line and the others join together
        return sequence

class TableLoader:
    def __init__(self, driver):
        self.driver = driver

    def get_catalytic_res_names(self):
        aa_catalycs = self.driver.find_elements_by_xpath("//table[@class='ll-template-table']/thead/tr[2]/td")[1:-1] ## Crop it from empty columns (first and last)
        aa_catalycs = [item.find_element_by_tag_name("button").text.split(",") for item in aa_catalycs] ## Names of residues in characters arrays
        return aa_catalycs

    def get_enzymeMinerQuery_names(self):
        datalist = self.driver.find_element_by_id("ll-manual-accessions")
        options = datalist.find_elements_by_xpath(".//option")
        return [i.get_attribute("value") for i in options]

    def get_catalytic_pos(self):
        trs = self.driver.find_elements_by_xpath("//table[@class='ll-template-table']/tbody/tr")
        aa_cat_rows = list(map(lambda a: [i.get_attribute("value") for i in a], [tr.find_elements_by_tag_name("input") for tr in trs]))
        aa_cat_dict = {}
        for row in aa_cat_rows:
             aa_cat_dict[row[0]] = list(map(lambda x: str(x) if x != '' else -1, row[1:]))
        return aa_cat_dict

    def select_queries(self, cat_dict=None):
        """Select sequences without empty catalytic residues sides as queries for further filtering"""
        if cat_dict == None:
            cat_dict = self.get_catalytic_pos()
        queries_dict = {}
        for k in cat_dict.keys():
            if cat_dict[k].count(-1) == 0:
                queries_dict[k] = cat_dict[k]
        return queries_dict

    def _debug_elem(self, elem):
        if isinstance(elem, list):
            print('Len of list elements is:', len(elem))
            if(len(elem) == 0):
                return
            else:
                elem = elem[0]
        text = elem.text
        value = elem.get_attribute('value')
        inner = elem.get_attribute('innerHtml')
        id = elem.get_attribute('id')
        print('Value:', value)
        print('innerHtml:',inner)
        print('Text:', text)
        print('ID:', id)

class CatalyticMSAPreprocessor:
    """
    Class for preprocessing of MSA from Pfam with preservation of catalytic residues
    in MSA based on EnzymeMiner.
    """
    def __init__(self, setuper):
        self.setuper = setuper
        self.ec = setuper.ec

    def proc_msa(self):
        self._enzymeMiner_handler() ##Setup queries and catalytic residues
        msa, references = self._pfam_handler()
        self._filter_against_ref(references, msa)

    def _enzymeMiner_handler(self):
        print("="*60)
        print("Scraping sequences from EnzymeMiner with EC number", self.ec)
        ## Connect to page with desired ec number
        page = PageHandler(ec=self.ec)
        try:
            ## Get tabs and switch their visibility
            page.select_all()
            page.switch_tabs()
            ## Select data from table
            self.queries = page.get_queries() ## uniprot names and positions of catalytic residues
            self.aa_cat_res = page.get_catalytic_res()
        finally:
            print("Closing connection to EnzymeMiner browser")
            page.close()

    def _pfam_handler(self, ):
        print("=" * 60)
        print("Translation of query names to Pfam name format")
        ## Connect to uniprot
        page = PageHandler()
        pfam_names = []
        uniprot_seqs = {}
        try:
            ## Get tabs and switch their visibility
            for k in self.queries.keys():
                pfam_id = page.translate(uniCode=k)
                pfam_names.append(pfam_id)
                uniprot_seqs[pfam_id] = (page.get_uniprot_seq(uniCode=k), self.queries[k]) ## Tuple(Uniprot original sequences, catalytic residues position)
                                                                                            # stored under pfam id in dictionary
        finally:
            print("Closing connection to Uniprot browser")
            page.close()
        ## Find pfam alignment in dataset
        msa = MSA(setuper=self.setuper, processMSA=False).load_msa()
        pfam_sequences = {}
        for k in msa.keys():
            for p in pfam_names:
                if p in k:
                    pfam_sequences[p] = msa[k]
                    # from Bio import SeqIO
                    # records = SeqIO.parse(self.setuper.MSA_fld+'PF00561_full.txt', "stockholm")
                    # count = SeqIO.write(records, self.setuper.MSA_fld+"THIS_IS_YOUR_OUTPUT_FILE.fasta", "fasta")
                    # print(k)
                    pfam_names.remove(p)
        ## Align those which were not founf in original dataset with pfam alignment
        if len(list(pfam_sequences.values())) > 0:
            key = next(iter(pfam_sequences))
            al = Alignmer(self.setuper.MSA_fld, pfam_sequences[key], key=key)
            for n in pfam_names:
                name, seq = al.align(uniprot_seqs[n][0], n)
                pfam_sequences[name] = seq
        ## TODO alignment to found sequences and kept of correct position against pfam alignment
        references = {}
        for k in pfam_sequences.keys():
            references[k] = self._allocate_catalytic_sites(uniprot_seqs[k][0], uniprot_seqs[k][1], pfam_sequences[k])
        return msa, references

    def _filter_against_ref(self, refs, msa):
        '''Look at given positions if residues fits catalytic residues from EnzymeMiner'''
        query_names = refs.keys()
        filtered_msa = {}
        for pfam_n in msa.keys():
            if pfam_n.split('/')[0] in query_names:
                ## Keep query sequences in MSA
                filtered_msa[pfam_n] = msa[pfam_n]
                continue
            ## Check sequence if has the catalytic residues same at least as one query
            for pos in refs.values():
                hit_cnt = len(self.aa_cat_res)
                for i, p in enumerate(pos):
                    if msa[pfam_n][p] in self.aa_cat_res[i]:
                        hit_cnt -= 1
                if hit_cnt <= 3: ## Match of residues in 100 percent of cases
                    filtered_msa[pfam_n] = msa[pfam_n]
                    break
        if self.setuper.stats:
            print('Catalytic residues query based filtering left:', len(filtered_msa.keys()))
        return filtered_msa

    def _allocate_catalytic_sites(self, uni_seq, sites_in_uni, pfam_seq):
        '''Return positions of catalytic sites in pfam MSA'''
        clear_pfam_seq = pfam_seq.replace('-', '').replace('.', '')
        idx = [] ## index in pfam MSA sequence
        for t in [i for i in range(0, len(pfam_seq))]:
            if pfam_seq[t] not in ['.', '-']:
                idx.append(t)
        cat_idx = [] ## Store positions of catalytic residues in MSA
        for site in [int(i) for i in sites_in_uni]:
            pos = site
            size = 2
            while size > 0:
                catalytic = uni_seq[pos - (size+1):pos + size] ## Take sourroundings 2 AA
                found_catalytic = re.findall('{}+'.format(catalytic), clear_pfam_seq)
                if len(found_catalytic) == 0:
                    size -= 1
                else:
                    break
            if size == 0:
                print('Catalytic residue', catalytic, ' on position', pos, ' was not found...')
                continue
            ## only first occurance take to account
            cat_idx.append(idx[clear_pfam_seq.index(found_catalytic[0])+size])
            # startIndex = clear_pfam_seq.index(found_catalytic[0])
            # endIndex = startIndex + len(found_catalytic[0]) - 1
            # print('-'*70)
            # print('catalytic residue pfam:', clear_pfam_seq[startIndex:endIndex])
            # print('uniprot catalytic', catalytic)
            # print('Residue pfam', pfam_seq[cat_idx[-1]], ' Residue expected', self.aa_cat_res[len(cat_idx)-1], ' Residue original', uni_seq[pos-1])
            # print('index by idx', idx[clear_pfam_seq.index(found_catalytic[0])+size], ' or ', clear_pfam_seq[clear_pfam_seq.index(found_catalytic[0])+size])
        return cat_idx

class Alignmer:
    def __init__(self, path, ref, key):
        self.file_name = path + 'tmp.fasta'
        self.ref_seq = ref
        self.key = key

    def align(self, uni_seq, name):
        fasta_dict = {}
        fasta_dict[self.key] = self.ref_seq
        fasta_dict[name] = uni_seq
        ## Now transform sequences back to fasta
        with open(self.file_name, 'w') as file_handle:
            for seq_name, seq in fasta_dict.items():
                file_handle.write(">" + seq_name + "\n" + seq + "\n")

        reference = 0
        count = 0
        with tempfile.NamedTemporaryFile() as temp:
            for record in SeqIO.parse(self.file_name, "fasta"):
                if count == 0:
                    reference = record
                else:
                    SeqIO.write([reference, record], temp.name, "fasta")
                    mafft_cline = MafftCommandline(input=temp.name)
                    stdout, stderr = mafft_cline()
                    aligned_fasta = stdout.split('>')[2]
                    seq_name = aligned_fasta.split()[0]
                    seq_seq = aligned_fasta.split()[1]
                count += 1
        return seq_name, seq_seq


if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = CatalyticMSAPreprocessor(tar_dir)
    msa.proc_msa()