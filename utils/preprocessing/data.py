from ast import literal_eval
from typing import List, Any
import json
import numpy as np

# trim the number from the ICPC code and return the alphabetical categories
def extract_icpc_categories(codes: Any, remove_admin=True) -> List[str]:
    print(codes)
    codes = literal_eval(codes)
    codes = [c[0].strip().upper() for c in codes]
    if remove_admin:
        codes = [c for c in codes if c != '-']
    return codes

def segment_without_overlapping(tokenizer, sents, maximum_length):
    '''
    used to segment descriptions or transcripts into chunks
    '''
    chunks = ['']
    prev =[]
    
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        if len(prev)+len(tokens) < maximum_length:
            chunks[-1] += " "+ sent
            prev += tokens

        else:
            chunks.append(sent)
            prev = tokens
    if '' in chunks:
        chunks.remove('')
    return chunks

def split_by_overlapping(slide_window, sents):
    '''
    split a list of sentences into chunks using sliding windows.
    '''
    examples = []
    for i in range(slide_window, len(sents)):
        start = max(i-5, 0)
        end = min(i+5, len(sents))
        example = sents[start:end]
        examples.append(example)
    return examples


def write_path(save_path, obj):
    with open(save_path,'w') as f:
        json.dump(obj, f, ensure_ascii=False)


def masking(examples, labels, label2name, tokenizer, prompt, max_length=512):
    '''
    used to do MLM task
    '''
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    for i, example in enumerate(examples):
        encoded_input = tokenizer(example, prompt.format('[MASK]'), padding='max_length', truncation=True, max_length=max_length)
        encoded_target = tokenizer(example, prompt.format(label2name[labels[i]]), padding='max_length', truncation=True, max_length=max_length)
        input_ids = encoded_input['input_ids'] 
        target_ids = encoded_target['input_ids']
        attention_mask.append(encoded_input['attention_mask'])
        token_type_ids.append(encoded_input['token_type_ids'])

        rands = np.random.random(len(input_ids))
        source, target = [], []
        # for r, t in zip(rands, input_ids):
        for ii in range(len(input_ids)):
            r, t = rands[ii], input_ids[ii]
            # maksed class name
            if t == tokenizer.mask_token_id:
                source.append(t)
                target.append(target_ids[ii])
            elif r < 0.15 * 0.8:
                source.append(tokenizer.mask_token_id)
                target.append(t)
            elif r < 0.15 * 0.9:
                source.append(t)
                target.append(t)
            elif r < 0.15:
                source.append(np.random.choice(tokenizer.vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(-100)
        all_input_ids.append(source)
        all_targets.append(target)

    return {'input_ids': all_input_ids, 
            'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
            'targets': all_targets}

def NSP(examples, labels, polarity, label2name, tokenizer, prompt, max_length=512, fine_grained=False):
    '''
    examples: list, training examples
    labels: list, training labels
    polarity: list, '1': postive, '0': negative
    fined_grained: fine_grained class names are multiple tokens
    '''
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    for i, example in enumerate(examples):
        # print('exmaple',example)
        if fine_grained:
            encoded_input = tokenizer(example, prompt.format(labels[i]), padding='max_length', truncation=True, max_length=max_length)
        else:
            encoded_input = tokenizer(example, prompt.format(label2name[labels[i]]), padding='max_length', truncation=True, max_length=max_length)
        all_input_ids.append(encoded_input['input_ids'])
        attention_mask.append(encoded_input['attention_mask'])
        token_type_ids.append(encoded_input['token_type_ids'])
        all_targets.append(int(polarity[i]))

    return {'input_ids': all_input_ids, 
            'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
            'targets': all_targets}


def labelmapping(labels, label_map):
    labels = [label_map[l] for l in labels] 
    return {'targets': labels}

def prediction_encoding(examples, tokenizer, prompt, max_length=512, withoutmask=False):
    '''used for prediction with MLM and conventional BERT classifier
        examples: nested list [[segment1, segment2,...],...]
        tokenizer: BERT tokenizer
        withoutmask: is used for conventional BERT classifier
    '''
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    for segments in examples:
        for seg in segments:
            if withoutmask:
                encoded_input = tokenizer(seg, padding='max_length', truncation=True, max_length=max_length)
            else:
                encoded_input = tokenizer(seg, prompt.format('[MASK]'), padding='max_length', truncation=True, max_length=max_length)
            input_ids = encoded_input['input_ids']
            all_input_ids.append(input_ids)
            targets = np.array(input_ids, copy=True)
            targets[targets != tokenizer.mask_token_id] = -100
            all_targets.append(targets)
            attention_mask.append(encoded_input['attention_mask'])
            token_type_ids.append(encoded_input['token_type_ids'])
    return {'input_ids': all_input_ids,'attention_mask': attention_mask, 'token_type_ids': token_type_ids,'targets': all_targets}


def binary_predictiton_encoding(examples, tokenizer, prompt, class_names, max_length=512):
    '''
    used for binary prediction in NSP task
    '''
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    for segments in examples:
        for seg in segments:
            for label in class_names: # iterate all class names for each example
                encoded_input = tokenizer(seg, prompt.format(label), padding='max_length', truncation=True, max_length=max_length)
                all_input_ids.append(encoded_input['input_ids'])
                # fake labels
                all_targets.append(0)
                attention_mask.append(encoded_input['attention_mask'])
                token_type_ids.append(encoded_input['token_type_ids'])

    return {'input_ids': all_input_ids,'attention_mask': attention_mask, 'token_type_ids': token_type_ids,'targets': all_targets}
            


