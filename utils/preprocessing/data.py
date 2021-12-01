from ast import literal_eval
from typing import List, Any
import json
import numpy as np

# trim the number from the ICPC code and return the alphabetical categories
def extract_icpc_categories(codes: Any, remove_admin=True) -> List[str]:
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
        # print('tokens', len(tokens))
        if len(prev)+len(tokens) < maximum_length:
            chunks[-1] += " "+ sent
            prev += tokens

        else:
            # print('length', len(tokens))
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
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    print(len(examples))
    for i in range(len(examples)):
        example = examples[i]
    # encoded_input = tokenizer(example['description'], prompt.format('[MASK]'), padding='max_length', truncation=True)
        encoded_input = tokenizer(example, prompt.format('[MASK]'), padding='max_length', truncation=True, max_length=max_length)
        encoded_target = tokenizer(example, prompt.format(label2name[labels[i]]), padding='max_length', truncation=True, max_length=max_length)
        input_ids = encoded_input['input_ids'] 
        target_ids = encoded_target['input_ids']
        attention_mask.append(encoded_input['attention_mask'])
        token_type_ids.append(encoded_input['token_type_ids'])

        rands = np.random.random(len(input_ids))
        source, target = [], []
        # for r, t in zip(rands, input_ids):
        for i in range(len(input_ids)):
            r, t = rands[i], input_ids[i]
            # maksed class name
            if t == tokenizer.mask_token_id:
                source.append(t)
                target.append(target_ids[i])
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

def labelmapping(labels, label_map):
    labels = [label_map[l] for l in labels] 
    return {'labels': labels}

def prompt_encoding(examples, tokenizer, prompt, max_length=512):
    all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
    all_segments = []
    for segments in examples:
        for seg in segments:
            encoded_input = tokenizer(seg, prompt.format('[MASK]'), padding='max_length', truncation=True, max_length=max_length)
            input_ids = encoded_input['input_ids']
            all_input_ids.append(input_ids)
            targets = np.array(input_ids, copy=True)
            targets[targets != tokenizer.mask_token_id] = -100
            all_targets.append(targets)
            attention_mask.append(encoded_input['attention_mask'])
            token_type_ids.append(encoded_input['token_type_ids'])
    return {'input_ids': all_input_ids,'attention_mask': attention_mask, 'token_type_ids': token_type_ids,'targets': all_targets}

# def normal_encoding(examples, tokenizer, prompt, max_length=512):
#     all_input_ids, all_targets, attention_mask, token_type_ids = [], [], [],[]
#     for example in examples:
#         encoded_input = tokenizer(example, prompt.format('[MASK]'), padding='max_length', truncation=True, max_length=max_length)
#         input_ids = encoded_input['input_ids']
#         all_input_ids.append(input_ids)

def softmax(logits):
    normalized = np.exp(logits - np.max(logits, axis = -1, keepdims=True))
    return normalized / np.sum(normalized, axis=-1, keepdims=True)

def merge_predictions(record_cnt, predictions):
        cum_sum = np.cumsum(record_cnt)
        assert cum_sum[-1] == len(predictions)
        final_predictions = np.zeros((len(record_cnt), predictions.shape[1]))
        prev = 0
        for i in range(len(cum_sum)):
            entry = cum_sum[i]
            final_predictions[i] = np.any(predictions[prev:entry], axis=0).astype(int)
            prev = entry
        return final_predictions

