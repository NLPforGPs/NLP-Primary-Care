## Results

### Results added by Edwin on 18th March 2022

This represents the current and most complete set of results. The previous
section is retained below as it includes a few extra details. 

#### Methods

Conventional BERT = A pretrained model (PubMedBERT-abstract-fulltext) with a linear layer on top.

Masked language model (MLM) prompting = A pretrained model (PubMedBERT-abstract) is given part of the transcript as input followed by a prompt. The prompt contains a missing word, which the model must guess. The prompt is: 'This is a problem of {}.' The word in '{}' is the class label that is to be predicted by the model. 

Next sentence prediction (NSP) prompting = A pretrained model (PubMedBERT-abstract) is used as a binary classifier for each possible label. For each label, it is given two inputs: part of the transcript and a prompt sentence as in MLM with the label filled in, e.g., 'This is a problem of Musculo-skeletal.'

For all three methods, the transcripts are split into chunks of up to 490 tokens. Each chunk is then treated as a separate data point. We merge the predictions for a whole transcript by predicting any labels that were assigned to any chunks in the transcript. For the probabilities, we take the maximum probability that each label received for any chunk in the transcript, then re-normalise. Conventional BERT and MLM are therefore multiclass classifiers for each chunk, but could output multiple labels across a long transcript. NSP is a binary classifier for each label.


#### Distant Supervision, Coarse-grained Topics

The goal of this experiment is to predict the top-level ICPC codes.
Only the CKS topics are available as training data for the classifiers,
and medical and custom stopwords are used.


| models                                           | F1-score | ROC-AUC | F1-score excluding the 'General' class | ROC-AUC excluding 'General' |
|--------------------------------------------------|----------|---------|----------|---------|
| Multi-class Naïve Bayes                          | 0.34 | 0.78 | 0.36 | 0.79 |
| Binary Naïve Bayes                               | 0.18 | 0.75 | 0.19 | 0.78 |
| Multi-class SVM                                  | 0.36 | 0.83 | 0.39 | 0.86 |
| Binary SVM                                       | 0.00 | 0.84 | 0.00 | 0.86 |
| Nearest centroid                                 | 0.36 | 0.64 | 0.39 | 0.65 |
| Conventional BERT Classifier                     | 0.53     | 0.55    | | |
| Masked Language Model (MLM)                      | **0.54** | 0.87    | | |
| Next Sentence Prediction (NSP) Prompting         | 0.42     | 0.87    | | |

We also tested different combinations of stopwords and different
sets of descriptions as the training text (CKS topics, ICPC code descriptions, both)
for all the non-neural network methods. In all cases, CKS topics were the best
training descriptions, and medical and custom stopwords give the best performance.
The table below shows F1 scores with different sets of stopwords with CKS topics for training:


| models                       | m,c | m | c | e |
|------------------------------|----------|---------|----------|---------|
| Multi-class Naïve Bayes      | 0.34 | 0.11 | 0.08 | 0.12 |
| Binary Naïve Bayes           | 0.18 | 0.00 | 0.01 | 0.00 |
| Multi-class SVM              | 0.36 | 0.04 | 0.03 | 0.04 |
| Nearest centroid             | 0.36 | 0.04 | 0.03 | 0.04 |


#### Distant Supervision, Fine-grained Topics

The goal of this experiment is to predict the CKS topics. We do not have gold
labels for this mapping, so we cannot evaluate it directly. Instead, we map the
predicted CKS topics to corresponding ICPC codes, and compare these with the
gold ICPC codes.

Only the CKS topics are available as training data for the classifiers,
and medical and custom stopwords are used.


| models                                  | F1-score |
|-----------------------------------------|----------|
| Multi-class Naïve Bayes                 | 0.23 |
| Binary Naïve Bayes                      | 0.00 |
| Multi-class SVM                         | 0.36 |
| Nearest centroid                        | 0.36 |
| Conventional BERT classifier            | 0.45 |
| Next Sentence Prediction (NSP) Prompting| 0.26 |

#### Supervision with Transcripts, Coarse-grained Topics

The goal of this experiment is to predict the top-level ICPC codes. We use
5-fold cross validation, where the transcripts are split into fifths, then
five experiments are run. In each experiment, 4/5 of the data is used to train
and 1/5 is used to test. The average F1 score and average ROC-AUC is given over
five runs.

Medical and custom stopwords are used.

| models                                           | F1-score | ROC-AUC |
|--------------------------------------------------|----------|---------|
| Multi-class Naïve Bayes                          | 0.30 | 0.77 |
| Binary Naïve Bayes                               | 0.28 | 0.77 |
| Multi-class SVM                                  | 0.18 | 0.68 |
| Binary SVM                                       | 0.16 | 0.77 |
| Nearest neighbours (k=3)                         | 0.16 | 0.59 |
| Nearest centroid                                 | 0.28 | 0.61 |


#### Stopwords + Supervision with Transcripts, Coarse-grained Topics

As above, except we test different combinations of stopwords and show
the best result.

| models                                           | F1-score | ROC-AUC | Stopwords |
|--------------------------------------------------|----------|-------- | --------- |
| Multi-class Naïve Bayes                          | 0.32 | 0.77 | m, c, e |
| Binary Naïve Bayes                               | 0.30 | 0.77 | m |
| Multi-class SVM                                  | 0.20 | 0.71 | c, e |  
| Binary SVM                                       | 0.18 | 0.77 | e |   
| Nearest neighbours (k=3)                         | 0.18 | 0.59 | m, c, e |
| Nearest centroid                                 | 0.30 | 0.62 | m, e |


-------------------------------------------------------------------------------

## Results added by Haishuo

> all these models using CKS descriptions to train, setups in the notebook and scripts.

#### Coarse-grained Results

| models                                           | F1-score | ROC-AUC |
|--------------------------------------------------|----------|---------|
| Naive Bayes Classifier                           | 0.34     | 0.78    |
| SVM (Support Vector Machine) Classifier          | 0.36     | 0.83    |
| Conventional BERT Classifier (original)          | **0.55** | 0.53    |
| Conventional BERT Classifier                     | 0.53     | 0.55    |
| Masked Language Model (MLM) Prompting (original) | 0.51     | 0.86    |
| MLM Prompting                                    | **0.54** | 0.87    |
| Next Sentence Prediction (NSP) Prompting         | 0.42     | 0.87    |

- Original MLM and conventional BERT classifier were trained on Colab with a larger batch size 16 and larger learning rate 1e-4 (GPU P100). Because the single 2080Ti Gpu memory is 11G, I reduce batch size and learning rate to 6 and 5e-5 respectively.
- For MLM method, apart from masking class names, randomly masking is adopted from BERT. It can improve performance since solely masking class names lead to overfitting.
- ROC-AUC is an approximated value. The maxium value of each category across different chunks are considered as the overall category probability for a complete transcript.

#### Fine-grained Results

| models                                  | F1-score |
|-----------------------------------------|----------|
| NB Classifier                           | 0.23     |
| SVM (Support Vector Machine) Classifier | 0.35     |
| Conventional BERT                       | 0.45     |
| Fine-grained NSP-1                      | 0.38     |
| Fine-grained NSP-2                      | 0.26     |


- Fine-grained NSP-1 represents directly using 16 categories to do prediction while Fine-grained NSP-2 represents predicting with health topics at first and merging it into 16 categories. NSP-2 takes longer time to get the results. Binary classification is done for each health topic (315 times) for each example.
- NSP F1-score is not as good as others since it predicts multiple labels for smaller chunks and they are merged for a transcript as a whole. This means this method has higher recall(0.79).
- ROC_AUC for fine-grained is not reported since you need to merge probabilities into 16 categories in many-to-many relationship.


- PLMs choosing: Prompting use PubMedBERT-abstract and conventional use PubMedBERT-abstract-fulltext
- Predicted label choosing: The above results are obtained by tagging the label with the highest probability as the predicted label.
