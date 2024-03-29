# coding: utf-8
'''
@filename:predict_plms.py
@Createtime: 2021/12/02 13:24:41
@author: Haishuo Fang
@description: This script is used to generate supportive chunks for model decision
'''


import torch
from nn_model import DescClassifier
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForSequenceClassification
import os
import numpy as np
import json
from utils.preprocessing.data import segment_without_overlapping
from torch.utils.data import DataLoader
from utils.utils import select_supportive_sentence, save_to_file
import argparse
from tqdm import tqdm
from oneinamillion.resource import PCC_BASE_DIR
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--label_path', type=str, default='label2id.json')
    parser.add_argument('--write_file', type=str, default='support_setences.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_prompt', type=bool, default=False, action='store_true')
    parser.add_argument('--interactive', type=bool, default=False, action='store_true')

    parser.add_argument('--predict_dir', type=str, default='./data/transcripts')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

    args = parser.parse_args()

    # path
    predict_data_dir = os.path.join(PCC_BASE_DIR, args.predict_dir)
    data_dir = os.path.join(PCC_BASE_DIR, args.data_dir)

    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
    label2name = {'A':'general', 'B':'blood', 'D':'digestive','F':'eye', 'H':'ear', 'K':'cardiovascular','L':'musculoskeletal',
    'N':'neurological','P':'psychological','R':'respiratory','S':'skin','T':'endocrine', 'U':'urological','W':'pregnancy','X':'female',
    'Y':'male'}
    class_names = list(label2name.values())


    with open(os.path.join(data_dir, args.label_path), 'r') as f:
        label2id = json.load(f)
    id2label = {label2id[label]: label2name[label] for label in label2id}
    print('id2label',id2label)

    if args.use_prompt:
        pre_model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
        
    else:
        config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2name))
        pre_model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
        
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    classifier = DescClassifier(pre_model, epochs=0, learning_rate=0)
    if device == torch.device('cpu'):
        print('using cpu...')
        checkpoint = torch.load(os.path.join(
                    args.save_dir, args.save_name + 'best-val-acc-model.pt'), map_location=device)
    else:
        print('using gpu...')
        checkpoint = torch.load(os.path.join(
                    args.save_dir, args.save_name + 'best-val-acc-model.pt'))

    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.to(device)
    if args.interactive:
        while True:

            input_text = input('Please input text:')
            # input_text= r"how are you sir?i'm not too bad at the moment actually.right, okay good.i just thought i'd come and touch base with you.a few weeks ago i had these little flares didn't i?that just flares up, and my hands aren't brilliant this morning but generally, because i'm taking the 5mg of the- the prednisolone.yes.that makes such a big difference, i think that was- okay.yes.obviously i saw professor last name about three weeks ago.yes.and she obviously outlined the drawbacks of taking prednisolone.but unfortunately the surgery that we were supposed to go to with mr last name, they booked me into the wrong surgery so i've got to go back.because i haven't had a lung function or anything.so i right.and how do things feel to you on that side?not too bad actually.no they're not too bad.i think, because i'm not doing so much exertion anymore, you don't- obviously stairs are a problem.but generally, i mean this has been coming for quite a long time so you get used to that, don't you?i always thought it was just because i was getting older, and obviously i'm a bit heavier than i used to be, but generally i'm not too bad actually.okay, what about mood side effects?yes, mood, i don't think that's ever really been- i don't think i ever really get to deal with it, do you know what i mean?obviously the tablets help.they keep my mood on a reasonable level, but i'm not very happy if you see what i mean.and i think that's more to do with the fact that i'm kind of stuck.right.does that make sense?yes, yes."
            if input_text == 'exit':
                break
            chunks = segment_without_overlapping(tokenizer, input_text, maximum_length=50)
            if args.use_prompt:
                prompt = checkpoint['prompt']
                print('prepare data')
                prompts = [prompt]*len(chunks)
                batch = tokenizer(chunks, prompts, padding=True, truncation=True, return_tensors='pt')

            else:
                batch = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')

            print('predicting...')
            predictions, probs = classifier.predict(batch, device, tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(class_names))
            # print(logits)

            res = select_supportive_sentence(probabilities=probs, chunks=chunks, class_names=np.array(class_names), threshold=0)
            # interative_map(logits, class_names, chunks)
        # heat_map(logits, class_names)
    else:
        dataset = load_dataset('./oneinamillionwrapper/transcript_evaldataset.py', download_mode="force_redownload", data_dir= predict_data_dir)

        all_probs, total_chunks = [], []

        for item in tqdm(dataset['transcripts'][:200]):
            chunks = segment_without_overlapping(tokenizer, item, maximum_length=50)
            if args.use_prompt:
                prompt = checkpoint['prompt']
                prompts = [prompt]*len(chunks)
                batch = tokenizer(chunks, prompts, padding=True, truncation=True, return_tensors='pt')
            else:
                batch = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')

            # print('predicting...')
            predictions, probs = classifier.predict(batch, device, tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(class_names))

            all_probs.extend(probs.tolist())
            total_chunks.extend(chunks)

        assert len(total_chunks) == len(all_probs)
        print('total_chunks', len(total_chunks))
        res = select_supportive_sentence(probabilities=np.array(all_probs), chunks=total_chunks, class_names=np.array(class_names), threshold=0)
        save_to_file(res, os.path.join(data_dir, args.write_file))