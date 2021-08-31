## Project Description

The project proposal asks the question: ‘Is it possible to build a proof-of-concept NLP tool to successfully “code” and
summarise a clinical primary care consultation from an audio transcription of that consultation?’. This work answers
this question.

## Folder Structure
```
NLP_GP
├───oneinamillion
│   ├───clinical_codes
│   ├───primary_care
├───oneinamillionwrapper
└───utils
    └───preprocessing
```


| package              | sub-package    | description                                                               |
|----------------------|----------------|---------------------------------------------------------------------------|
| oneinamillion        |                | class for getting clinical consultation datapairs                         |
|                      | clinical_codes | class for getting ICPC or CKS codes                                       |
|                      | primary_care   | contains sub-package classes to parse transcripts/ gp records             |
| oneinamillionwrapper |                | Wrapper class for clinical datapairs to be used with HuggingFace datasets |
| utils                |                | Commonly used methods                                                     |
|                      | preprocessing  | methods for data preprocessing                                            |

## Development Setup

1. Mount the network drive for the
   [One In a Million archive](https://www.bristol.ac.uk/primaryhealthcare/researchthemes/one-in-a-million/)
   - On a windows computer: From windows Explorer access, or Map a network drive, to:
     `\\rdsfcifs.acrc.bris.ac.uk\NLP_One_In_A_Million`

   - On a Mac: In the Finder select 'Go->Connect To Server...' from the menu (or Command-K) and enter 
     `smb://rdsfcifs.acrc.bris.ac.uk/NLP_One_In_A_Million` into the dialogue box.
     

2. Point the program to use the mounted network drive
   either:
   - Set environment variables on linux/ macos, e.g.
     ```
     export PCC_BASE_DIR='~/mounted_location'
     ```
   - or update `resources.py` in `oneinamillion` package, e.g.
      ```
      # Insert line to override variables
      PCC_BASE_DIR = '~/mounted_location'
      ```
   - for available variable names, look at `resources.py`.
   

3. create new 
   [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
   for project

   ```
   # create environment using the requirements file
   conda env create --file nlp_gp.yml
   
   #activate environment
   conda activate NLP_GP
   ```


4. launch jupyter lab/ notebook and launch python notebook, 
   either from the command line, or from the GUI launcher.
   [Jupyter Lab guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html).

   ```
   jupyter lab
   # OR
   jupyter notebook
   ```

## Updating Source Datasets
**Prerequisites**
You must install AntiWord to process .doc documents, which is only available on [Windows](https://www.softpedia.com/get/Office-tools/Other-Office-Tools/Antiword.shtml).

By default, the primary care data-pair parser (i.e. `PCConsultation` in `pc_consultation.py`)
would read from the cache. To inform the parser about changes in the consultation documents,
the cache must be cleared.

To do this, simply replace
```
orig_dataset = parser.get_pd()
```
in block two of the notebook with,
```
orig_dataset = parser.get_pd(from_raw=True)
```

Make sure to do this *once* only and *undo the changes* after a single execution, 
as parsing and preparing raw consultation documents is a costly process.

### FAQ

**Missing consultations but located in Drive**

There might also be various reason that the parser fails to read consultation data pairs, e.g. 
incorrect file naming, unsupported file format (pdf).
You may modify the behaviour of the parser by editing the respective classes in
`patient_record.py` and `transcript.py`.

There might also be missing documents for that consultation.
There are three documents required for every primary care consultation record.
By default, the parser would only return samples only if the three documents are present.
i) the transcript, ii) the gp record, iii) the clinical codes for the record.

