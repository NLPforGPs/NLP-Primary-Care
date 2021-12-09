## Project Description


The project proposal asks the question: ‘Is it possible to build a proof-of-concept NLP tool to successfully “code” and
summarise a clinical primary care consultation from an audio transcription of that consultation?’. This work answers
this question.

- [Project Description](#project-description)
- [Folder Structure](#folder-structure)
- [Development Setup](#development-setup)
- [Updating Source Datasets](#updating-source-datasets)
- [Fine-tune Pretrained Models with Distant Supervision](#fine-tune-pretrained-models-with-distant-supervision)
  - [Setup](#setup)
    - [Initialise repo in your home directory](#initialise-repo-in-your-home-directory)
    - [Copy RDSF file to workplace](#copy-rdsf-file-to-workplace)
    - [Create log directory](#create-log-directory)
  - [GO Run it!](#go-run-it)
  - [Results](#results)
    - [NSP Dataset Generation](#nsp-dataset-generation)
    - [To Do](#to-do)

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



## Fine-tune Pretrained Models with Distant Supervision
> This method is to fine-tune the pretrained models using descriptions and adapt to transcripts.

### Setup
#### Initialise repo in your home directory
```
git clone https://github.com/NLPforGPs/NLP-Primary-Care
```
#### Copy RDSF file to workplace
```
cp -r /projects/NLP_One_In_A_Million /user/work/username/
```

Then, you need to set `/user/work/username/NLP_One_In_A_Million` as the 'PCC_BASE_DIR' env variable.

#### Create log directory

 Run `. ./scripts/create_logdir.sh` at first. Results will be written in this dir.

### GO Run it!

- Train a classifier
   -  MLM(multi-class)
      ```
      sbatch ./scripts/train/train-mlm.sh
      ```
   - NSP (binary)

      ```
      sbatch ./scripts/train/train-nsp.sh
      ``` 

   - Conventional BERT classifier
      ```
      sbatch ./scripts/train/train-conventional.sh
      ```
   - Fine-grained categories
      ```
      sbatch ./scripts/train/train-conventional-fine-grained.sh
      ```

- Evaluate performance using transcripts
  -  MLM(multi-class)
      ```
      sbatch ./scripts/test/test-mlm.sh
      ```
   - NSP (binary)

      ```
      sbatch ./scripts/test/test-nsp.sh
      ``` 

   - Conventional BERT classifier
      ```
      sbatch ./scripts/test/test-conventional.sh
      ```
   - Fine-grained categories
      ```
      sbatch ./scripts/test/test-conventional-fine-grained.sh
      ```

   More details could be found in `./scripts/`.

### Results
> all these models using CKS descritpions to train, setups in the notebook and scripts.


| models             | F1-score    |                            ROC-AUC         |
|----------------------|----------------|---------------------------------------------------------------------------|
| Navie Bayes Classifier | 0.34 | 0.78 |
| SVM Classifier | 0.36 | 0.83 |
| Conventional BERT Classifier (original)| **0.55** | 0.53
| Conventional BERT Classifier| 0.50 | 0.53
| MLM Prompting (original)| 0.51 | 0.86|
| MLM Prompting| **0.54** | 0.87|
| NSP Prompting| 0.42| 0.87|
| Fine-grained Conventional|  0.45| - |
| Fine-grained NSP| | - |

- Original is trained on Colab with larger batch size 16 and larger learning rate 1e-4 (GPU P100).
- Prompting use PubMedBERT-abstract and conventional use PubMedBERT-abstract-fulltext
- Because the single 2080Ti Gpu memory is 11G, I reduce batch size and learning rate to 8 and 5e-5, respectively.
- NSP F1-score is not as good as others since it predicts multiple labels for smaller chunks and they are merged for a transcript as a whole. This means this method has higher recall(0.79).
- ROC-AUC is an approximated value. The maxium value of each category across different chunks are considered as the overall probability for a complete transcript.

#### NSP Dataset Generation
It is implemented in `generate_binary_descriptions`(`prepare_data.py`). 
- some health topics are mapped to multiple categories. When sampling negative examples, it will avoid sampling the same descriptions in other categories.
- The dataset is balanced.
- This dataset is based on multi-class datasets


#### Fine-grained Dataset Geneartion
- There are over 400 health topics but when it is splitted into train and dev dataset, only topics with long sentences which can be divided into train/test datasets are retained, 319 topics in total.


#### To Do

- This structure is not the best structure. It could be improved like investigating if different datasets could be merged in one universal Dataset Class, like SQUAD.
- Hyperparameters could be tuned. It showed different batch sizes and learning rate can affect the model performance. However, deep learning methods are more stable than traditional methods in this task.



# FQA

**Missing consultations but located in Drive**

There might also be various reason that the parser fails to read consultation data pairs, e.g. 
incorrect file naming, unsupported file format (pdf).
You may modify the behaviour of the parser by editing the respective classes in
`patient_record.py` and `transcript.py`.

There might also be missing documents for that consultation.
There are three documents required for every primary care consultation record.
By default, the parser would only return samples only if the three documents are present.
i) the transcript, ii) the gp record, iii) the clinical codes for the record.

