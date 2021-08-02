# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The ‘One in a Million’study: creating a database of UK primary care consultations"""

from dataclasses import dataclass

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
from oneinamillion.pc_consultation import PCConsultation

_CITATION = """\
@article{jepson2017one,
  title={The ‘One in a Million’study: creating a database of UK primary care consultations},
  author={Jepson, Marcus and Salisbury, Chris and Ridd, Matthew J and Metcalfe, Chris and Garside, Ludivine and Barnes, Rebecca K},
  journal={British Journal of General Practice},
  volume={67},
  number={658},
  pages={e345--e351},
  year={2017},
  publisher={British Journal of General Practice}
}
"""

_DESCRIPTION = """\
the Bristol Archive Project aimed to create an archive of high quality video recordings of primary care consultations,
plus linked survey and electronic patient records data with permissions in place for reuse by other bonafide researchers
and for future research and training purposes.
"""

_HOMEPAGE = "https://www.bristol.ac.uk/primaryhealthcare/researchthemes/one-in-a-million/"

# TODO: Add the licence for the dataset here
_LICENSE = ""

FEATURES = datasets.Features(
    {
        "record_id": datasets.Value("string"),
        "pt_records": datasets.Sequence(
            {
                "date": datasets.Value("string"),
                "gp": datasets.Value("string"),
                "additional": datasets.Value("string"),
                "allergy": datasets.Value("string"),
                "assessment": datasets.Value("string"),
                "comment": datasets.Value("string"),
                "document": datasets.Value("string"),
                "examination": datasets.Value("string"),
                "family_history": datasets.Value("string"),
                "follow_up": datasets.Value("string"),
                "history": datasets.Value("string"),
                "lab_results": datasets.Value("string"),
                "medication": datasets.Value("string"),
                "problem": datasets.Value("string"),
                "procedure": datasets.Value("string"),
                "referral": datasets.Value("string"),
                "regime_review": datasets.Value("string"),
                "result": datasets.Value("string"),
                "social": datasets.Value("string"),
                "template_entry": datasets.Value("string"),
                "test_request": datasets.Value("string"),
                "x_ray": datasets.Value("string"),
            }
        ),
        "transcript": {
            "start_datetime": datasets.Value("string"),
            "duration": datasets.Value("string"),
            "conversation": datasets.Sequence(
                {
                    "speaker": datasets.Value("string"),
                    "dialogue": datasets.Value("string"),
                }
            )
        },
    }
)


@dataclass
class OneInAMillionBuilderConfig(datasets.BuilderConfig):
    n_fold: int = 1
    ith_fold: int = 1


class OneInAMillion(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = OneInAMillionBuilderConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # These kwargs will be passed to _generate_examples
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(
            self, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        pc_reader = PCConsultation()

        train_split, test_split = [], []
        for ii, (train_split, test_split) in enumerate(pc_reader.create_train_test_split(self.config.n_fold)):
            if ii + 1 == self.config.ith_fold:
                break

        if split == 'train':
            selected_idx = train_split
        else:
            selected_idx = test_split

        for ii, data in enumerate(pc_reader.get_sequence(selected_idx)):

            pt_records = []
            for record in data.pt_record.records:
                dct = vars(record)
                date = dct.pop('date')
                dct['date'] = str(date)
                pt_records.append(dct)

            yield ii, {
                "record_id": data.pt_record.id,
                "pt_records": pt_records,
                "transcript": {
                    "start_datetime": str(data.transcript.start_datetime),
                    "duration": str(data.transcript.duration),
                    "conversation": [
                        {
                            "speaker": c[0],
                            "dialogue": c[1],
                        }
                        for c in data.transcript.conversations
                    ]
                }
            }
