import os.path
from os import environ

"""
You must point the program to the correct location for the RDSF NLP One in a Million drive.

You may:
a)
Set PCC_BASE_DIR to the mounted network location, and have other files relatively loaded.

b)
Alternatively, you may set PCC_CODES_DIR, PCC_PT_RECORD_RAW_DIR, PCC_TRANSCRIPT_RAW_DIR, PCC_PREPARED_PATH
and/or PCC_TRANSCRIPT_DIR (optional), PCC_PT_RECORD_DIR (optional)
"""

PCC_BASE_DIR = environ.get('PCC_BASE_DIR') or r"Z:/"

# Clinical Codes
PCC_CODES_DIR = environ.get('PCC_CODES_DIR') or os.path.join(PCC_BASE_DIR, 'codes')
PCC_CODES_FILE = os.path.join(PCC_CODES_DIR, 'OiAM_ICPC-2.xlsx')
# ICPC
PCC_ICPC_DIR = os.path.join(PCC_CODES_DIR, 'ICPC')
PCC_ICPC_FILE = "icpc-2e-v7.0.csv"
# CKS
# ...

# Raw source documents
PCC_PT_RECORD_RAW_DIR = environ.get('PCC_PT_RECORD_RAW_DIR') or os.path.join(PCC_BASE_DIR, 'Patient records')
PCC_TRANSCRIPT_RAW_DIR = environ.get('PCC_TRANSCRIPT_RAW_DIR') or os.path.join(PCC_BASE_DIR, r'Transcripts\transcripts')

# Prepared documents
PCC_PREPARED_PATH = environ.get('PCC_PREPARED_PATH') or os.path.join(PCC_BASE_DIR, 'prepared')
PCC_TRANSCRIPT_DIR = environ.get('PCC_TRANSCRIPT_DIR') or os.path.join(PCC_PREPARED_PATH, 'transcripts')
PCC_PT_RECORD_DIR = environ.get('PCC_PT_RECORD_DIR') or os.path.join(PCC_PREPARED_PATH, 'records')
