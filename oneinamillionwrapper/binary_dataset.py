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
    name = "BinaryDescriptionDataset"
    version = datasets.Version("1.0.0")

FEATURES = datasets.Features(
    {
    "description": datasets.Value("string"),
    "codes": datasets.Value("string"),
    "polarity": datasets.Value("string")
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

        print(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    'filepath': os.path.join(self.config.data_dir, 'train.json')

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
            yield ii, {'description': item[0], 'codes': item[1], 'polarity': item[2]}