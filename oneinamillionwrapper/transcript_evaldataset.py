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
from typing import List
import itertools
@dataclass
class BuilderConfig(datasets.BuilderConfig):
    # def __init__(self, name, test_size, chunk_size, train_path, test_path, **kwargs):
        # super().__init__()
    name = "TranscriptEvalDataset"
    version = datasets.Version("1.0.0")

FEATURES = datasets.Features(
    {
    "transcript": datasets.Sequence(datasets.Value("string")),
    "codes": datasets.Sequence(datasets.Value("string")),
    "split_nums": datasets.Value("int32")
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': os.path.join(self.config.data_dir, 'transcript.json')
                    # "filepath": data_files['train']

                },
            )
        ]
    
    def _generate_examples(self, filepath):
        '''
        split into train and test set
        '''
        logging.info("Generating examples...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        for ii, item in enumerate(data):
            yield ii, {'transcript': item[0], 'codes': item[1], 'split_nums': len(item[0])}



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', model_max_length=512)
    # dataset = DescriptionDataset()
    dataset = load_dataset('/Users/hsfang/workspace/consolidation/NLP-Primary-Care/oneinamillionwrapper/transcript_evaldataset.py', 
    download_mode="force_redownload", data_dir='/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/transcripts')
    # load_dataset('json', data_files={'train': '/Users/hsfang/workspace/consolidation/NLP-Primary-Care/data/cks_only/490_train_CKS_only.json'})
    # test = load_dataset('DescriptionDataset')
    
    # data = dataset.map(lambda x: itertools.chain(x['transcripts']), batched=True)

    # dataloader = torch.utils.data.DataLoader(data, batch_size=32)
    print(next(iter(dataloader)))



        



