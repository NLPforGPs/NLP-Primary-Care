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
├── classification.ipynb
├── log                                         // used to save results
│   ├── test
│   └── train
├── nlp_gp.yml                                  // Conda Environment
├── nn_model                                    // Deep learning models   
│   ├── __init__.py
│   └── desc_classifier.py
├── oneinamillion                               // data extraction and preparation
│   ├── __init__.py
│   ├── clinical_codes
│   ├── common.py
│   ├── pc_consultation.py
│   ├── primary_care
│   └── resources.py
├── oneinamillionwrapper                        // Datasets for DL methods
│   ├── __init__.py
│   ├── binary_dataset.py
│   ├── description_dataset.py
│   ├── one_in_a_million.py
│   └── transcript_evaldataset.py
├── prepare_data.py                             // Split and Prepare descritpions and transcripts
├── requirements_conda.txt
├── requirements_pip.txt
├── run_plms.py                                 // Entrypoint of DL models
├── scripts                                     // Scripts for BP
│   ├── create_logdir.sh
│   ├── test
│   └── train
└── utils                                       // useful tools for other modules
    ├── __init__.py
    ├── metrics
    ├── preprocessing
    ├── stopwords
    ├── transcripts.py
    └── utils.py
```


| package              | sub-package    | description                                                               |
|----------------------|----------------|---------------------------------------------------------------------------|
| oneinamillion        |                | package for getting clinical consultation datapairs                         |
|                      | clinical_codes | class for getting ICPC or CKS codes                                       |
|                      | primary_care   | contains sub-package classes to parse transcripts/ gp records             |
| oneinamillionwrapper |                | Datasets(transcripts and descripts) used in deep learning models using HuggingFace datasets|
|                      | binary_dataset | Used for NSP method |
|                      | description_dataset| Used for MLM and conventional classifier |
|                      | transcript_evaldataset| Used for model evaluation |
| utils                |                | Commonly used methods                                                     |
|                      | preprocessing  | methods for data preprocessing                                            |
|                      | metrics        | package for model evaluation
| nn_model             |                | package for different deep learning models                                |


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
> This method is to fine-tune the pretrained models using descriptions and adapt them to transcripts.

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
```
vi ~/.bashrc
export PCC_BASE_DIR='/user/work/username/NLP_One_In_A_Million/'
source ~/.bashrc
```

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

### Parameters Explanation

- `--model_dir`, `--model_name` `--label_path` decide what model to use. `fg_label2id.json` is for fine-grained models while `label2id.json` is for conventional models. `origin_label2id.json` is for original conventional BERT classifier.
- To `--do_train` or `--do_predict` is a question
- `--use_mlm` or `--use_nsp` could be specified
- `--fine_grained_desc` decide whether to use health topics
- `--multi_data_path` is the data path for multiclas(conventional, mlm)and `--binary_data_path` for nsp method
- `--predict_data_dir` is often the path of `transcripts, which could be combined with chunk_size to generate new datasets.
- error anlysis is detailed in error analysis section




#### To Do

- This structure is not the perfect structure. It could be improved like investigating if different datasets could be merged in one universal Dataset Class, like SQUAD. How to manage different models and datasets is what I am still learning.

- 'Z' should be removed in original spreadsheet(I aovid it in the programme but delete it physically will be better to avoid incorrect file generation).

- Hyperparameters could be tuned. It showed different batch sizes and learning rate can affect the model performance. I split descritpionts dataset into train and dev datasets to tune heyperparameters rather than using limited transcripts to tune hyperparmaeters. It seems to be out-of-distribution generalisation.

- deep learning methods are more stable than traditional methods in this task.

### Error Analysis and HeatMap
It should be noted that when the chunk size gets smaller, the merged F1-score would be lower. This is because we just take the category with the highest probability as the predicted label of each chunk no matter how low it is. So i think  smaller chunks could be set for analysis and larger chunk for evluation.

Interesting finding: F1 score of Conventional BERT classifier descreses more dramatically than that of MLM when chunk size get smaller. I think it is corresponding to auc-roc

#### Usage
`sbatch ./scripts/test/error_analysis.sh`
- use `--model_dir`,`--model_name`,``--label_path` to specify a model.
- use `--fine_grained_desc` to use health topics.
- `--chunk_size` is used to generate examples shorter than the chunk size if not exist. `--predict_data_dir "transcripts-50"` specifies predict_data path
- `--ea_file` specifies name of the output file. It is in the

#### Generated Files
It can be found in `/Volumes/NLP_One_In_A_Million/prepared/dl_data/error_analysis/`. There are two files `error_analysis-50.xls` and `error_analysis-490.xls` which are generated using conventional BERT with different chunk sizes.



# FAQ

**Missing consultations but located in Drive**

There might also be various reason that the parser fails to read consultation data pairs, e.g.
incorrect file naming, unsupported file format (pdf).
You may modify the behaviour of the parser by editing the respective classes in
`patient_record.py` and `transcript.py`.

There might also be missing documents for that consultation.
There are three documents required for every primary care consultation record.
By default, the parser would only return samples only if the three documents are present.
i) the transcript, ii) the gp record, iii) the clinical codes for the record.
