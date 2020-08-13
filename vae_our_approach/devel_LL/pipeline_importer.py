import os

if 'PIP_BRANCH' in os.environ:
    if os.environ['PIP_BRANCH'] == 'True':
        from catalytic_res_msa_filter import CatalyticMSAPreprocessor as MSA
    else:
        from msa_prepar import MSA as MSA

    ## Unset linux environment variable
    del os.environ['PIP_BRANCH']
else:
    print("The os environment variable was not set!!")
    exit(1)
