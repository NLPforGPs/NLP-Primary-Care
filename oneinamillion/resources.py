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

PCC_BASE_DIR = environ.get('PCC_BASE_DIR') or r"/Volumes/OIAM_to_delete"  # NLP_One_In_A_Million"

ICPC_CKS_DESC = 'icpc_cks_description.csv'
CKS_DESC = 'cks_description.csv'
ICPC2CKS = 'icpc2cks.json'

DL_DATA = os.path.join(PCC_BASE_DIR, 'prepared/dl_data')

# Clinical Codes
PCC_CODES_DIR = environ.get('PCC_CODES_DIR') or os.path.join(PCC_BASE_DIR, 'codes')
# PCC_CODES_FILE = os.path.join(PCC_CODES_DIR, 'OiAM_ICPC-2.xlsx')
PCC_CODES_FILE = os.path.join(PCC_CODES_DIR, 'one_line_per_problem.xls')
PCC_CODES_LINK_FILE = "icpc2_to_cks.xlsx"
# ICPC
PCC_ICPC_DIR = os.path.join(PCC_CODES_DIR, 'ICPC')
PCC_ICPC_FILE = "icpc-2e-v7.0.csv"
# CKS
PCC_CKS_DIR = os.path.join(PCC_CODES_DIR, 'CKS')
PCC_CKS_FILE = "cks_health_topics-29-07-2021.csv"

# Raw source documents
PCC_PT_RECORD_RAW_DIR = environ.get('PCC_PT_RECORD_RAW_DIR') or os.path.join(PCC_BASE_DIR, 'Patient records')
PCC_TRANSCRIPT_RAW_DIR = environ.get('PCC_TRANSCRIPT_RAW_DIR') or os.path.join(PCC_BASE_DIR, 'Transcripts','transcripts')

# Prepared documents
PCC_PREPARED_PATH = environ.get('PCC_PREPARED_PATH') or os.path.join(PCC_BASE_DIR, 'prepared')
PCC_TRANSCRIPT_DIR = environ.get('PCC_TRANSCRIPT_DIR') or os.path.join(PCC_PREPARED_PATH, 'transcripts')
PCC_PT_RECORD_DIR = environ.get('PCC_PT_RECORD_DIR') or os.path.join(PCC_PREPARED_PATH, 'records')
