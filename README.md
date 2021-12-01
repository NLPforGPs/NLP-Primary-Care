## Project Description


The project proposal asks the question: ‘Is it possible to build a proof-of-concept NLP tool to successfully “code” and
summarise a clinical primary care consultation from an audio transcription of that consultation?’. This work answers
this question.

- [Project Description](#project-description)
- [Folder Structure](#folder-structure)
- [Development Setup](#development-setup)
- [Updating Source Datasets](#updating-source-datasets)
- [Fine-tuning Pretrained Models with Distant Supervision](#fine-tuning-pretrained-models-with-distant-supervision)
  - [FAQ](#faq)

## Folder Structure
```
├── README.md
├── acknowledgements.md
├── classification.ipynb
├── nlp_gp.yml
├── nn_model
│   ├── __init__.py
│   └── desc_classifier.py                    // deep learning classifiers
├── oneinamillion                             // data extraction and preparation
│   ├── __init__.py
│   ├── __pycache__
│   ├── clinical_codes
│   ├── common.py
│   ├── pc_consultation.py
│   ├── primary_care
│   └── resources.py
├── oneinamillionwrapper                       // Dataset for deep learning model
│   ├── __init__.py
│   ├── description_dataset.py                 // Load description dataset using dataset API
│   ├── one_in_a_million.py                     
│   └── transcript_evaldataset.py              // Load transcript dataset using dataset API
├── prepare_data.py                            // Prepare and split transcript data and descritpion data  
├── requirements_conda.txt
├── requirements_pip.txt
├── run_plms.py                                 // entrypoint for run deep learning models
└── utils                                       // Tools used in other components
    ├── __init__.py
    ├── metrics
    ├── preprocessing
    ├── stopwords
    └── transcripts.py
```


| package              | sub-package    | description                                                               |
|----------------------|----------------|---------------------------------------------------------------------------|
| oneinamillion        |                | package for getting clinical consultation datapairs                         |
|                      | clinical_codes | class for getting ICPC or CKS codes                                       |
|                      | primary_care   | contains sub-package classes to parse transcripts/ gp records             |
| oneinamillionwrapper |                | Datasets(transcripts and descripts) used in deep learning models using HuggingFace datasets|
| utils                |                | Commonly used methods                                                     |
|                      | preprocessing  | methods for data preprocessing                                            |
| nn_model             |                | package for different deep learning models                                |
| metrics              |                | package for model evaluation                                              |


## Development Setup

1. Mount the network drive for the [One In a Million archive](https://www.bristol.ac.uk/primaryhealthcare/researchthemes/one-in-a-million/)

   If you are not on the university network, connect to the VPN.

   - On a windows computer: From windows Explorer access, or Map a network drive, to:
     `\\rdsfcifs.acrc.bris.ac.uk\NLP_One_In_A_Million`

   - On a Mac: In the Finder select 'Go->Connect To Server...' from the menu (or Command-K) and enter 
     `smb://rdsfcifs.acrc.bris.ac.uk/NLP_One_In_A_Million` into the dialogue box. The drive will be mounted to `/Volumes/NLP_One_In_A_Million/`.
   
   - On Linux (Ubuntu):
     
     replace *\<USERNAME\>* with your own UOB login username
     ```
     # before we can mount Windows shared drive, we need to install CIFS
     sudo apt install cifs-utils
     
     # create mount point and mount network drive
     sudo mkdir /mnt/nlp_one_in_a_million
     sudo mount -t cifs //rdsfcifs.acrc.bris.ac.uk/NLP_One_In_A_Million /mnt/nlp_one_in_a_million/ -o user=<USERNAME>,domain=UOB
     ```

3. Point the program to use the mounted network drive
   either:
   - Set environment variables on linux/ macos, e.g.
     ```
     export PCC_BASE_DIR='/mnt/nlp_one_in_a_million'
     ```
   - or update `resources.py` in `oneinamillion` package, e.g.
      ```
      # Insert line to override variables
      PCC_BASE_DIR = '/mnt/nlp_one_in_a_million'
      ```
   - for available variable names, look at `resources.py`.
   

3. create new 
   [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
   for project.
   
   To install CUDA with PyTorch on **macOS**, please refer to
   [guide](https://pytorch.org/get-started/locally/#mac-from-source).

   ```
   # navigate to project directory
   cd NLP-GP/
   
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



## Fine-tuning Pretrained Models with Distant Supervision
> This method is to fine-tune the pretrained models using descriptions and adapt to transcripts

- Train a classifier using descriptions
   -  Prompt
      -  running on local machine
         ```
         python3 ./run_plms.py --batch_size 8 --do_train True --epoch 15 --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --learning_rate 1e-4 --weight_decay 1e-4 --use_prompt True --model_dir models/prompt --model_name multiclass-abstract-optimal --multi_class True --train_data_dir "dl_data/desc/CKS only" --prompt "This is a problem of {}."
         ```
      - running on Bluepebble
         ```
         sbatch ./scripts/train/train-prompt.sh
         ```

   - Conventional(turn off `argument prompt`, change model to PubMedBERT(abstract-fulltext))
      - running on local machine
         ```
         python3 ./run_plms.py --batch_size 32 --do_predict True  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --use_prompt True --load_checkpoint True --multi_class True --model_dir models/coventional --model_name multiclass-abstract-conventional --prompt 'This is a problem of {}.' --predict_data_dir 'dl_data/transcripts'
         ```
      - running on Bluepebble
         ```
         sbatch ./scripts/train/train-conventional.sh
         ```
   

- Evaluate performance using transcripts
   -  Prompt
      -  running on local machine

         ```
         python3 ./run_plms.py --batch_size 32 --do_predict True  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --use_prompt True --load_checkpoint True --multi_class True --model_dir models/prompt --model_name multiclass-abstract-modified --prompt 'This is a problem of {}.' --predict_data_dir 'dl_data/transcripts'
         ```
      - running on Bluepebble
         ```
         sbatch ./scripts/test/test-prompt.sh
         ```
   - Conventional
      - running on local machine
         ```
         python3 ./run_plms.py --batch_size 32 --do_predict True  --pretrained_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --load_checkpoint True --multi_class True --model_dir models/coventional --model_name multiclass-abstract-modified --prompt 'This is a problem of {}.' --predict_data_dir 'dl_data/transcripts'
         ```
      - running on Bluepebble
         ```
         sbatch ./scripts/test/test-conventional.sh
         ```
   More details could be found in `./scripts/``


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

