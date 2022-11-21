from datasets import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.special import expit
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
import os
from torch import tensor
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig, AutoModelForMaskedLM, \
    BertForNextSentencePrediction
from nn_model import DescClassifier
from oneinamillion.resources import PCC_BASE_DIR
import torch

from prepare_data import generate_binary_per_class
from utils.preprocessing.data import NSP, segment_without_overlapping
from nltk import tokenize
from utils.utils import merge_predictions, stratified_multi_label_split


# pretrained_model = 'prajjwal1/bert-tiny'  # 'albert-base-v2'
pretrained_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'


class NSPDataset(Dataset):
    def __init__(self, text_and_polarities, labels, split_nums, pretrained_model, prompt, id2label):
        self.text, self.polarities = zip(*text_and_polarities)
        self.text = np.array(self.text)
        self.polarities = np.array(self.polarities).astype(int)
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=True)
        self.split_nums = split_nums
        self.prompt = prompt
        self.max_len = 512
        self.id2label = id2label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        encoding = self.tokenizer.encode_plus(
            self.text[i],
            self.prompt.format(label2name[self.id2label[self.labels[i]]]),
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        targets = tensor(self.polarities[i], dtype=torch.long)
        masked_input = encoding['input_ids'].flatten()

        return {
            'text': self.text[i],
            'input_ids': masked_input,
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': targets,
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }


class StandardClassificationDataset(Dataset):
    def __init__(self, raw_text, labels, split_nums, pretrained_model, prompt, id2label):
        self.text = raw_text
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=True)
        self.split_nums = split_nums
        self.max_len = 512

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        encoding = self.tokenizer.encode_plus(
            self.text[i],
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        targets = tensor(self.labels[i] if self.labels is not None else -100, dtype=torch.long)
        masked_input = encoding['input_ids'].flatten()

        return {
            'text': self.text[i],
            'input_ids': masked_input,
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': targets,
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }


class MLMClassifierDataset(Dataset):
    def __init__(self, raw_text, labels, split_nums, pretrained_model, prompt, id2label):
        self.text = raw_text
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, local_files_only=True)
        self.split_nums = split_nums
        self.prompt = prompt
        self.max_len = 512
        self.id2label = id2label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        input_encoding = self.tokenizer.encode_plus(
            self.text[i], self.prompt.format('[MASK]'),
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = input_encoding['input_ids'].flatten()

        if self.labels is None:  # MLM classifier in prediction mode
            masked_input = input_ids
            targets = np.array(input_encoding['input_ids'].flatten(), copy=True)
            targets[targets != self.tokenizer.mask_token_id] = -100
        else:  # MLM classifier in training mode
            target_encoding = self.tokenizer.encode_plus(
                self.text[i],
                self.prompt.format(label2name[self.id2label[self.labels[i]]]),
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            target_ids = target_encoding['input_ids'].flatten()

            rands = np.random.random(len(target_ids))
            masked_input = []
            targets = []

            # for r, t in zip(rands, input_ids):
            for ii in range(len(target_ids)):
                r, t = rands[ii], input_ids[ii]
                # maksed class name
                if t == self.tokenizer.mask_token_id:
                    masked_input.append(t)
                    targets.append(target_ids[ii])
                elif r < 0.15 * 0.8:
                    masked_input.append(self.tokenizer.mask_token_id)
                    targets.append(t)
                elif r < 0.15 * 0.9:
                    masked_input.append(t)
                    targets.append(t)
                elif r < 0.15:
                    masked_input.append(np.random.choice(self.tokenizer.vocab_size - 1) + 1)
                    targets.append(t)
                else:
                    masked_input.append(t)
                    targets.append(-100)

        return {
            'text': self.text[i],
            'input_ids': np.array(masked_input),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'targets': np.array(targets),
            'token_type_ids': input_encoding['token_type_ids'].flatten(),
        }


def do_text_chunking(text, tokenizer, chunk_size, labels):
    chunks = []
    chunk_labels = []
    split_nums = []
    for i, doc in enumerate(text):
        sents = tokenize.sent_tokenize(doc)
        chunks_i = segment_without_overlapping(tokenizer, sents, chunk_size)
        chunk_labels_i = [labels[i]] * len(chunks_i)
        chunks.extend(chunks_i)
        chunk_labels.extend(chunk_labels_i)
        split_nums.append(len(chunks_i))

    return chunks, chunk_labels, split_nums


label2name = {'A':'general', 'B':'blood', 'D':'digestive', 'F':'eye', 'H':'ear', 'K':'cardiovascular',
              'L':'musculoskeletal', 'N':'neurological', 'P':'psychological', 'R':'respiratory', 'S':'skin',
              'T':'endocrine', 'U':'urological', 'W':'pregnancy', 'X':'female', 'Y':'male'}


def prepare_multiclass_training_set(chunks, chunk_labels, nclasses, dataset_constructor, training_mode, prompt, id2label):
    if training_mode != 'ICPC only':  # not enough data in ICPC to do this
        y_hot = np.zeros((len(chunks), nclasses))
        y_hot[range(len(chunks)), chunk_labels] = 1
        chunks_train, chunks_dev, labels_train, labels_dev = stratified_multi_label_split(chunks, y_hot, seed=20211125,
                                                                                          test_size=0.2)
        labels_train = np.argmax(labels_train, 1)
        labels_dev = np.argmax(labels_dev, 1)

        dataset_train = dataset_constructor(chunks_train, labels_train, None, pretrained_model, prompt, id2label)
        dataset_dev = dataset_constructor(chunks_dev, labels_dev, None, pretrained_model, prompt, id2label)
    else:
        dataset_train = dataset_constructor(chunks, chunk_labels, None, pretrained_model, prompt, id2label)
        dataset_dev = dataset_train

    return dataset_train, dataset_dev


def prepare_binary_training_set(chunks, chunk_labels, nclasses, dataset_constructor, training_mode, prompt, id2label):
    # if we are doing NSP, we need to create binary labels from the chunks next
    # raw_data is a list of pairs of [chunk, chunk_label]
    # processed_data has the same format plus the polarity label, [chunk, chunk_label, polarity]
    processed_chunks = generate_binary_per_class([list(tup) for tup in list(zip(chunks, chunk_labels))], {})
    chunks, chunk_labels, polarities = zip(*processed_chunks)
    chunks = list(zip(chunks, polarities))

    # now we can handle the data points as with the multiclass setup...
    return prepare_multiclass_training_set(chunks, chunk_labels, nclasses, dataset_constructor, training_mode, prompt, id2label)


def prepare_multiclass_test_set(chunks, chunk_labels, split_nums, nclasses, dataset_constructor, prompt, id2label):
    dataset_test = dataset_constructor(chunks, chunk_labels, split_nums, pretrained_model, prompt, id2label)
    return dataset_test


def prepare_binary_test_set(chunks, chunk_labels, split_nums, nclasses, dataset_constructor, prompt, id2label):

    output_chunks, output_labels, output_polarities = [], [], []
    for chunk in chunks:
        for l in range(nclasses):
            output_chunks.append(chunk)
            output_labels.append(l)
            output_polarities.append(0)
    output_chunks = list(zip(output_chunks, output_polarities))
    return prepare_multiclass_test_set(output_chunks, output_labels, split_nums, nclasses, dataset_constructor, prompt, id2label)


def run_bert_conventional(text_train, y_train, id2label, text_test, run_name, training_mode, trained_classifier=None):
    model_dir = os.path.join(PCC_BASE_DIR, 'models_edwin/conventional')
    model_name = 'full-text-conventional_' + run_name

    print('using traditional bert classifier...')

    nclasses = y_train.shape[1]

    config = BertConfig.from_pretrained(pretrained_model, num_labels=nclasses)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, config=config)

    return run_bert_classifier(text_train, y_train, id2label, text_test, training_mode, trained_classifier, model_dir,
                               model_name, model, prepare_multiclass_training_set, prepare_multiclass_test_set, False,
                               False, StandardClassificationDataset)


def run_nsp_classifier(text_train, y_train, id2label, text_test, run_name, training_mode, trained_classifier=None):
    model_dir = os.path.join(PCC_BASE_DIR, 'models_edwin/mlm')
    model_name = 'mlm-abstract-5e-5_' + run_name

    print('using next sentence prediction bert classifier...')

    model = BertForNextSentencePrediction.from_pretrained(pretrained_model)

    prompt = "This is a problem of {}."

    return run_bert_classifier(text_train, y_train, id2label, text_test, training_mode, trained_classifier, model_dir,
                               model_name, model, prepare_binary_training_set, prepare_binary_test_set,
                               False, True, NSPDataset, prompt)


def run_mlm_classifier(text_train, y_train, id2label, text_test, run_name, training_mode, trained_classifier=None):
    model_dir = os.path.join(PCC_BASE_DIR, 'models_edwin/mlm')
    model_name = 'mlm-abstract-5e-5_' + run_name

    print('using masked language model bert classifier...')

    model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    prompt = "This is a problem of {}."

    return run_bert_classifier(text_train, y_train, id2label, text_test, training_mode, trained_classifier, model_dir,
                               model_name, model, prepare_multiclass_training_set, prepare_multiclass_test_set, True,
                               False, MLMClassifierDataset, prompt)


def run_bert_classifier(text_train, y_train, id2label, text_test, training_mode, trained_classifier, model_dir, model_name, model,
                        prepare_training_set, prepare_test_set, use_mlm, use_nsp, dataset_constructor, prompt=None):

    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
    if training_mode == 'ICPC only':
        epochs = 1 # 5  # we cannot create a dev split for early stopping, so lower number of epochs to avoid overfitting.
    else:
        epochs = 1 # 10
    weight_decay = 1e-4
    batch_size = 8
    stop_epochs = 3
    chunk_size = 490
    load_checkpoint = False

    nclasses = y_train.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)  # , local_files_only=True)

    if trained_classifier is not None:
        classifier = trained_classifier  # just reuse it without training again
    else:
        model.to(device)
        classifier = DescClassifier(model=model, epochs=epochs, learning_rate=1e-5, weight_decay=weight_decay)

        # create a dataset object from the data
        chunks, chunk_labels, split_nums = do_text_chunking(text_train, tokenizer, chunk_size, np.argmax(y_train, 1))
        dataset_train, dataset_dev = prepare_training_set(chunks, chunk_labels, nclasses, dataset_constructor, training_mode, prompt, id2label)

        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False)

        print('Training...')
        classifier.train(train_loader=train_dataloader, dev_loader=dev_dataloader, save_dir=model_dir,
                         save_name=model_name, stop_epochs=stop_epochs, device=device, prompt=prompt,
                         load_checkpoint=load_checkpoint, ckpt_name=model_name)

    # format test data
    chunks, chunk_labels, split_nums = do_text_chunking(text_test, tokenizer, chunk_size, len(text_test) * [0])
    # don't pass in the labels to the prediction dataset...
    dataset_test = prepare_test_set(chunks, None, split_nums, nclasses, dataset_constructor, prompt, id2label)

    predict_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    if device == torch.device('cpu'):
        checkpoint = torch.load(os.path.join(model_dir, model_name + '_best-val-acc-model.pt'), map_location=device)
    else:
        checkpoint = torch.load(os.path.join(model_dir, model_name + '_best-val-acc-model.pt'))
    classifier.load_state_dict(checkpoint['state_dict'])

    print('Predicting...')
    id2name = np.array([label2name[l] for l in id2label])
    predictions, pred_probs = classifier.predict(predict_dataloader, device, tokenizer, use_mlm=use_mlm,
                                                 use_nsp=use_nsp,
                                                 class_names=id2name)
    # convert to one-hot encodings
    predictions = np.array(predictions).flatten()
    y_hot = np.zeros((len(chunks), nclasses))
    if not np.issubdtype(predictions.dtype, np.number):
        predictions = [np.argwhere(id2name == p)[0][0] for p in predictions]
    y_hot[range(len(chunks)), predictions] = 1

    # merge labels for each transcript
    final_predictions = merge_predictions(dataset_test.split_nums, np.array(y_hot))
    final_probs = merge_predictions(dataset_test.split_nums, np.array(pred_probs), probs=True)

    print('Completed training BERT classifier.')
    return final_predictions, final_probs, classifier


def run_multiclass_naive_bayes(X_train, y_train, X_test):
    # Each transcript can have one label.

    clf = MultinomialNB(alpha=0.001, fit_prior=False)
    clf.fit(X_train, np.argmax(y_train, 1))

    y_pred_probs = clf.predict_proba(X_test)

    nmissing_classes = y_train.shape[1] - y_pred_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_pred_probs.shape[0], nmissing_classes))
        y_pred_probs = np.concatenate((y_pred_probs, y_missing), axis=1)

    y_pred_mat = y_pred_probs > 0.1

    return y_pred_mat, y_pred_probs


def run_binary_naive_bayes(X_train, y_train, X_test):
    # Consider applying each label to every transcript, i.e., a binary decision per code

    nclasses = y_train.shape[1]

    y_pred_mat = np.zeros((X_test.shape[0], nclasses))
    y_pred_probs = np.zeros((X_test.shape[0], nclasses))

    for c in range(nclasses):
        # iterate over the classes and make a classifier for each class

        clf = MultinomialNB(alpha=0.001)
        clf.fit(X_train, y_train[:, c])

        y_te_pred_c = clf.predict(X_test)
        # y_pred_mat[:, c] = y_te_pred_c

        y_pred_probs[:, c] = clf.predict_proba(X_test)[:, 1]
        y_pred_mat[:, c] = y_pred_probs[:, c] > 0.1

    return y_pred_mat, y_pred_probs


def run_multiclass_svm(X_train, y_train, X_test):
    nclasses = y_train.shape[1]

    clf = SVC(kernel='rbf', C=2)

    clf.fit(X_train, np.argmax(y_train, 1))

    y_pred = clf.predict(X_test)
    y_pred_probs = clf.decision_function(X_test)

    nmissing_classes = y_train.shape[1] - y_pred_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_pred_probs.shape[0], nmissing_classes))
        y_pred_probs = np.concatenate((y_pred_probs, y_missing), axis=1)

    y_pred_mat = np.zeros((y_pred.shape[0], nclasses))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    return y_pred_mat, y_pred_probs


def run_binary_svm(X_train, y_train, X_test):
    nclasses = y_train.shape[1]

    y_pred_mat = np.zeros((X_test.shape[0], nclasses))
    y_pred_probs = np.zeros((X_test.shape[0], nclasses))

    for c in range(nclasses):
        # iterate over the classes and make a classifier for each class

        clf = SVC(kernel='rbf', C=2)
        clf.fit(X_train, y_train[:, c])

        y_te_pred_c = clf.predict(X_test)

        y_pred_probs[:, c] = clf.decision_function(X_test)

        y_pred_mat[:, c] = expit(y_pred_probs[:, c]) > 0.1

    return y_pred_mat, y_pred_probs


def run_nearest_centroid(X_train, y_train, X_test):
    nc_clf = NearestCentroid(metric='euclidean')
    nc_clf.fit(X_train, np.argmax(y_train, 1))
    y_pred = nc_clf.predict(X_test)

    y_pred_mat = np.zeros((y_pred.shape[0], y_train.shape[1]))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    return y_pred_mat, y_pred_mat


def run_nearest_neighbors(X_train, y_train, X_test, n_neighbors=3):
    kn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
    kn_clf.fit(X_train, np.argmax(y_train, 1))

    y_pred = kn_clf.predict(X_test)
    y_pred_mat = np.zeros((y_pred.shape[0], y_train.shape[1]))
    y_pred_mat[np.arange(y_pred.shape[0]), y_pred] = 1

    y_probs = kn_clf.predict_proba(X_test)

    nmissing_classes = y_train.shape[1] - y_probs.shape[1]
    if nmissing_classes:
        y_missing = np.zeros((y_probs.shape[0], nmissing_classes))
        y_probs = np.concatenate((y_probs, y_missing), axis=1)

    return y_pred_mat, y_probs