import os

if ['PIP_CAT', 'PIP_PAPER', 'PIP_SCORE'] in os.environ:
    if os.environ['PIP_CAT'] == 'True':
        from catalytic_res_msa_filter import CatalyticMSAPreprocessor as MSA
    elif os.environ['PIP_PAPER'] == 'True' or os.environ['PIP_SCORE'] == 'False':
        # Default variant in the case of setting just --no_score_filter param
        from msa_prepar import MSA as MSA
    elif os.environ['PIP_SCORE'] == 'True':
        from msa_filter_scorer import MSAFilterCutOff as MSA

    ## Unset linux environment variable
    del os.environ['PIP_SCORE']
    del os.environ['PIP_CAT']
    del os.environ['PIP_PAPER']
else:
    print("The os environment variables were not set!!")
    exit(1)
