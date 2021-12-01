import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from dataset import DescLMDataset, BinaryDescLMDataset, BinaryIterDataset, DescDataset
from nn_model import DescClassifier
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from utils.metrics import evaluate_classifications
import numpy  as np
import os
import logging
import datasets
from datasets import load_dataset
from prepare_data import generate_descriptions, prepare_transcripts_eval
from utils.preprocessing.data import masking, labelmapping, prompt_encoding
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--stop_epochs', type=int, default=2)
    parser.add_argument('--pretrained_model', type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--train_datapath', type=str, default='./data/train.json')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--label_path', type=str, default='label2id.json')
    parser.add_argument('--dev_datapath', type=str, default='./data/test.json')
    parser.add_argument('--predict_dir', type=str, default='./data/splitted_transcripts.json')
    parser.add_argument('--test_datapath', type=str, default='./data/splitted_transcripts.json')
    parser.add_argument('--save_dir', type=str, default='./model')
    parser.add_argument('--save_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='This is a problem of {}.')
    parser.add_argument('--selected_mode', type=str, default='CKS only')
    parser.add_argument('--chunk_size', type=int, default=490)
    parser.add_argument('--max_length', type=int, default=512)
    

    parser.add_argument('--use_prompt', type=bool, default=False)
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--multi_class', type=bool, default=False)


    
    args = parser.parse_args()
    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))

    label2name = {'A':'general', 'B':'blood', 'D':'digestive','F':'eye', 'H':'ear', 'K':'cardiovascular','L':'musculoskeletal',
    'N':'neurological','P':'psychological','R':'respiratory','S':'skin','T':'endocrine', 'U':'urological','W':'pregnancy','X':'female',
    'Y':'male'}

    if args.use_prompt:
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    else:
        config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2name))
        # config.num_labels = len(label2name)
        # config.problem_type = 'single_label_classification'
        # model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model)
        model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)

    model.to(device)
    
    classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    # load data
    if not os.path.exists(args.data_dir):
        logging.info('data_dir not exists, generating data....')
        os.makedirs(args.data_dir)
        # this will generate a train and test datasets in args.data_dir
        generate_descriptions(tokenizer=tokenizer, chunk_size = args.chunk_size, test_size = 0.2, selected_mode=args.selected_mode, save_path = args.data_dir)


    if args.do_train:
        logging.info('training...')
        # using data loader script to load data
        dataset = load_dataset('./oneinamillionwrapper/description_dataset.py', download_mode="force_redownload", data_dir= args.data_dir)

        if args.use_prompt:
            print('using prompt methods...')
            # using masking method to generate prompt
            dataset = dataset.map(lambda e: masking(e['description'], e['codes'], label2name, tokenizer, args.prompt), batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
            dev_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

            # test_dataset = dataset['test'].map(lambda e: masking(e['description'], tokenizer, args.prompt), batched=True)
            # test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])

        else:
            print('using traditional bert classifier...')
            labels = list(set(dataset['train']['codes']))
            label2id = {key: ix for ix, key in enumerate(labels)}
            print('label2id',label2id)
            with open(os.path.join(args.data_dir, args.label_path), 'w') as f:
                json.dump(label2id, f)
            dataset = dataset.map(lambda e: labelmapping(e['codes'], label2id), batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
            dev_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

        
        classifier.train(train_loader=train_dataloader, dev_loader=dev_dataloader, save_dir=args.save_dir, save_name=args.save_name, stop_epochs=args.stop_epochs, device=device, prompt=args.prompt, load_checkpoint=args.load_checkpoint, ckpt_name=args.ckpt_name)

    if args.do_predict:
        print('Predicting...')
        if not os.path.exists(args.predict_dir):
            logging.info('data_dir not exists, generating data....')
            os.makedirs(args.predict_dir)
        # this will generate a train and test datasets in args.data_dir
            prepare_transcripts_eval(tokenizer=tokenizer, max_length= args.chunk_size, save_path = args.predict_dir)
        
        # predict_data = load_data(args.predict_dir)
        if args.load_checkpoint:
            print('path',os.path.join(
                args.save_dir, args.save_name+'best-val-acc-model.pt'))
            if device == torch.device('cpu'):
                checkpoint = torch.load(os.path.join(
                    args.save_dir, args.save_name+'best-val-acc-model.pt'), map_location=device)
            else:
                checkpoint = torch.load(os.path.join(
                    args.save_dir, args.save_name+'best-val-acc-model.pt'))
            classifier.load_state_dict(checkpoint['state_dict'])

        # load data
        dataset = load_dataset('./oneinamillionwrapper/transcript_evaldataset.py', download_mode="force_redownload", data_dir= args.predict_dir)
        dataset = dataset['test']
        splited_nums = dataset['split_nums']
        print('spliteed_nums', splited_nums)

        # predict_loader = DataLoader(BinaryDescLMDataset(data=predict_data['all_trans'], prompt=prompt, label2name=label2name, pretrained_model= args.pretrained_model, do_train=False, random_mask=False), batch_size=args.batch_size, shuffle=False)
        if args.use_prompt:
            print('Prompt based method.....')
            encoded_dataset = dataset.map(lambda e: prompt_encoding(e['transcript'], tokenizer, args.prompt, args.max_length), batched=True, remove_columns=['transcript', 'codes', 'split_nums'])
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            
            # encoded_dataset = dataset.map(lambda e: prompt_encoding(e['transcript'], tokenizer, args.prompt), batched=True, remove_columns=dataset.column_names)
            predict_dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)
            id2label = None

        else:
            print('conventional classifier...')
            with open(os.path.join(args.data_dir, args.label_path), 'r') as f:
                label2id = json.load(f)
            id2label = {label2id[label]: label2name[label] for label in label2id}
            print('id2label',id2label)
            encoded_dataset = dataset.map(lambda e: tokenizer(e['all_segments'], padding=True, truncation=True, max_length=args.max_length), batched=True, remove_columns=dataset.column_names)
            encoded_dataset = encoded_dataset.map(lambda e: {'targets': len(e['input_ids'])*[0]}, batched=True)
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            predict_dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)

        class_names = list(label2name.values())
        predictions, _ = classifier.predict(predict_dataloader, device, tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(class_names))
        # predictions,_ = classifier.predict(predict_dataloader, device, pos_id=tokenizer.convert_tokens_to_ids('really'), neg_id = tokenizer.convert_tokens_to_ids('not'), tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(list(label2name.values())))
        # plot_heatmap(class_logits, predict_data['splited_nums'], class_names)

        mult_lbl_enc = MultiLabelBinarizer()
        y_hot = mult_lbl_enc.fit_transform(dataset['codes'])
        predictions = mult_lbl_enc.transform(predictions)

        final_predictions = merge_predictions(predict_data['splited_nums'], np.array(predictions))
        
        print(evaluate_classifications(y_hot, final_predictions, list(label2name.values()), show_report=True))