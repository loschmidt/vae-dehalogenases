__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/10 11:30:00"

from download_MSA import Downloader
from pipeline import StructChecker
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains

from time import sleep
from PIL import Image
from io import BytesIO

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
        self._enzymeMiner_handler() ##Setup queries and catalytic residues
        self._pfam_handler()

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
            self.queries = page.get_queries() ## names and positions of catalytic residues
            self.aa_cat_res = page.get_catalytic_res()
        finally:
            print("Closing connection to EnzymeMiner browser")
            page.close()

    def _pfam_handler(self):
        print("=" * 60)
        print("Translation of queries names to Pfam name format")
        ## Connect to uniprot
        page = PageHandler()
        pfam_names = []
        try:
            ## Get tabs and switch their visibility
            for k in self.queries.keys():
                pfam_names.append(page.translate(uniCode=k))
        finally:
            print("Closing connection to Uniprot browser")
            page.close()
        print(pfam_names)

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = CatalyticMSAPreprocessor(tar_dir)