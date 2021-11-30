import datasets
from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
import logging
import numpy as np
import os
import json
import torch
from utils.preprocessing.data import masking

@dataclass
class BuilderConfig(datasets.BuilderConfig):
    # def __init__(self, name, test_size, chunk_size, train_path, test_path, **kwargs):
        # super().__init__()
    name = "DescriptionDataset"
    version = datasets.Version("1.0.1")

FEATURES = datasets.Features(
    {
    "description": datasets.Value("string"),
    "codes": datasets.Value("string")
    }
)

class DescriptionDataset(datasets.GeneratorBasedBuilder):
    # BUILDER_CONFIGS = BuilderConfig(name="DescriptionDataset", test_size=0.2, chunk_size=490, train_path='./train.json', test_path='./test.json')
    BUILDER_CONFIG_CLASS = BuilderConfig
    

    def _info(self):
        return datasets.DatasetInfo(

            features=FEATURES,
            supervised_keys=None,
        )


    def _split_generators(self, dl_manager):
        # These kwargs will be passed to _generate_examples
        # data_path = {'train': '/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/cks_only/490_train_CKS_only.json', 
        # 'test': '/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/cks_only/490_test_CKS_only.json'}
        # dfiles = dl_manager.download_and_extract(data_files)
        # print(self.config.data_files['train'][0])
        print(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    'filepath': os.path.join(self.config.data_dir, 'train.json')
                    # "filepath": data_files['train']

                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": os.path.join(self.config.data_dir, 'test.json')
                },
            )
        ]
    
    def _generate_examples(self, split, filepath):
        '''
        split into train and test set
        '''
        logging.info("Generating examples...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        for ii, item in enumerate(data):
            yield ii, {'description': item[0], 'codes': item[1]}



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', model_max_length=512)
    # dataset = DescriptionDataset()
    dataset = load_dataset('/Users/hsfang/workspace/consolidation/NLP-Primary-Care/oneinamillionwrapper/description_dataset.py', 
    download_mode="force_redownload", data_dir='/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/cks_only')
    # load_dataset('json', data_files={'train': '/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/cks_only/490_train_CKS_only.json'})
    # test = load_dataset('DescriptionDataset')
    print(dataset['test'])
    labels = list(set(dataset['train']['codes']))
    label2id = {key: ix for ix, key in enumerate(labels)}
    train_dataset = dataset['train'].map(lambda e: label2id[e['codes']], batched=True)
    print(train_dataset[0])
    exit()
    # data = test['test'].map(lambda x: masking(x['description'], tokenizer,'This is {}'), batched=True)

    dataloader = torch.utils.data.DataLoader(data, batch_size=32)
    print(next(iter(dataloader)))



        



