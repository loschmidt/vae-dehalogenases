import os

for var in ['PIP_CAT', 'PIP_PAPER', 'PIP_SCORE']:
    if var not in os.environ:
        print("The os environment variables were not set!! Variable : ", var)
        exit(1)
if os.environ['PIP_CAT'] == 'True':
    from msa_handlers.catalytic_res_msa_filter import CatalyticMSAPreprocessor as MSA
elif os.environ['PIP_PAPER'] == 'True' or os.environ['PIP_SCORE'] == 'False':
    # Default variant in the case of setting just --no_score_filter param
    from msa_handlers.msa_preparation import MSA as MSA
elif os.environ['PIP_SCORE'] == 'True':
    from msa_handlers.msa_preprocessor import MSAPreprocessor as MSA

## Unset linux environment variable
del os.environ['PIP_SCORE']
del os.environ['PIP_CAT']
del os.environ['PIP_PAPER']

