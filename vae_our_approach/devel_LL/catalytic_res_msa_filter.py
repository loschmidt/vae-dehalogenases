__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/10 11:30:00"

from download_MSA import Downloader
from pipeline import StructChecker
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options

from time import sleep

class CatalyticMSAPreprocessor:
    """
    Class for preprocessing of MSA from Pfam with preservation of catalytic residues
    in MSA based on EnzymeMiner.
    """
    def __init__(self, setuper):
        self.setuper = setuper
        self.EnzymeMiner = "https://loschmidt.chemi.muni.cz/enzymeminer/ec/"
        self.ec = setuper.ec
        self._enzymeMiner_handler()

    def _enzymeMiner_handler(self):
        print("="*60)
        print("Scraping sequences from EnzymeMiner with EC number", self.ec)

        ## Connect to page with desired ec number
        URL = self.EnzymeMiner + self.ec

        opts = Options()
        opts.headless = True
        assert opts.headless  # Operating in headless mode
        browser = Firefox(options=opts)
        browser.get(URL)
        try:
            print("Loading SwissProt tab page...")
            sleep(1) ## Just wait to load the page, stupid
            html_code = browser.page_source
            if 'btn btn-secondary' in html_code:
                print("Yes it is in")
            else:
                print("No it is not in")
            ## Get buttons and select desired one
            buttons = browser.find_elements_by_class_name("btn-secondary")
            ## Find button select all
            for btn in buttons:
                if btn.text == 'Select all':
                    btn.click()
                    print("Pressing select all button... ")
                    sleep(1)
                    break
            ## Get tabs and switch their visibility
            tabs = browser.find_elements_by_class_name("tab-pane")
            browser.execute_script("arguments[0].setAttribute('class',arguments[1])", tabs[0], 'tab-pane')
            browser.execute_script("arguments[0].setAttribute('class',arguments[1])", tabs[1], 'tab-pane active')
            print("Switching to custom sequences tab...")
            sleep(1)
            ## Select data from table
            mytable = browser.find_elements_by_id('manualOtherSequencesField')
            sleep(1)
            print(mytable.get_attribute("placeholder"))
            for row in mytable.find_elements_by_css_selector('tr'):
                for cell in row.find_elements_by_tag_name('td'):
                    print(cell.text)
            i = 0
            ##for b in rows:
             ##   i += 1
            res_page = browser.page_source
            if 'r1-0' in res_page:
                print("Yes it is in result")
            else:
                print("No it is not in result")

            ##pprint(results)
            ##print(results[0].text)
        finally:
            browser.close()


if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = CatalyticMSAPreprocessor(tar_dir)