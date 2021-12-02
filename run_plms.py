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
from utils.utils import merge_predictions
from torch.utils.data import DataLoader
from oneinamillion.resources import PCC_BASE_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--stop_epochs', type=int, default=2)
    parser.add_argument('--pretrained_model', type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--train_data_dir', type=str, default='/data', help='directory for train data')
    parser.add_argument('--label_path', type=str, default='label2id.json', help='path of saved label2id mapping for conventional classifier')
    parser.add_argument('--predict_data_dir', type=str, default='/data/transcript', help='directory for data to predict')
    parser.add_argument('--model_dir', type=str, default='./model', help='directory to save model')
    parser.add_argument('--model_name', type=str, default='test', help='name of the saved model')
    parser.add_argument('--ckpt_name', type=str, default='test', help='name of the loaded model')
    parser.add_argument('--prompt', type=str, default='This is a problem of {}.', help='prompt')
    parser.add_argument('--selected_mode', type=str, default='CKS only', help='choose to use which description')
    parser.add_argument('--chunk_size', type=int, default=490, help='used to split transcripts and descriptions into small chunks')
    parser.add_argument('--max_length', type=int, default=512, help='max length used in PLMs')
    

    parser.add_argument('--use_prompt', type=bool, default=False, action="store_true", help='donot include it unless you use it')
    parser.add_argument('--do_train', type=bool, default=False, action="store_true", help='do not include unless you use it')
    parser.add_argument('--do_predict', type=bool, default=False, action="store_true", help='do not include unless you use it')
    parser.add_argument('--load_checkpoint', type=bool, default=False, action="store_true", help='do not include unless you use it')
    parser.add_argument('--multi_class', type=bool, default=False, action="store_true", help='do not include unless you use it')


    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

    # training data directory, descriptions
    train_data_dir = os.path.join(PCC_BASE_DIR, args.train_data_dir)
    # prediction data directory, transcripts
    predict_data_dir = os.path.join(PCC_BASE_DIR, args.predict_data_dir)
    # model save path
    model_dir = os.path.join(PCC_BASE_DIR, args.model_dir)



    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
    # This map would be used in prompt to predict real word
    label2name = {'A':'general', 'B':'blood', 'D':'digestive','F':'eye', 'H':'ear', 'K':'cardiovascular','L':'musculoskeletal',
    'N':'neurological','P':'psychological','R':'respiratory','S':'skin','T':'endocrine', 'U':'urological','W':'pregnancy','X':'female',
    'Y':'male'}

    if args.use_prompt:
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    else:
        config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2name))
        model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)

    model.to(device)
    
    classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    if args.do_train:
        logging.info('training...')

        if not os.path.exists(train_data_dir): # prepare dataset for model training
            logging.info('data_dir not exists, generating data....')
            os.makedirs(train_data_dir)
            # this will generate a train and test datasets in data_dir
            generate_descriptions(tokenizer=tokenizer, chunk_size = args.chunk_size, test_size = 0.2, selected_mode=args.selected_mode, save_path = train_data_dir)

        # using data loader script to load data
        dataset = load_dataset('./oneinamillionwrapper/description_dataset.py', download_mode="force_redownload", data_dir= train_data_dir)

        if args.use_prompt:
            logging.info('using prompt methods...')
            # using masking method to generate prompt
            dataset = dataset.map(lambda e: masking(e['description'], e['codes'], label2name, tokenizer, args.prompt), batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
            dev_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

            # test_dataset = dataset['test'].map(lambda e: masking(e['description'], tokenizer, args.prompt), batched=True)
            # test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])

        else:
            logging.info('using traditional bert classifier...')
            labels = list(set(dataset['train']['codes']))
            label2id = {key: ix for ix, key in enumerate(labels)}
            print('label2id',label2id)
            with open(os.path.join(args.train_data_dir, args.label_path), 'w') as f:
                json.dump(label2id, f)
            dataset = dataset.map(lambda e: labelmapping(e['codes'], label2id), batched=True)
            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
            dev_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)

        
        classifier.train(train_loader=train_dataloader, dev_loader=dev_dataloader, save_dir=model_dir, model_name=args.model_name, stop_epochs=args.stop_epochs, device=device, prompt=args.prompt, load_checkpoint=args.load_checkpoint, ckpt_name=args.ckpt_name)

    if args.do_predict:
        logging.info('Predicting...')
        if not os.path.exists(predict_data_dir):
            logging.info('data_dir not exists, generating data....')
            os.makedirs(predict_data_dir)
        # this will generate a train and test datasets in args.data_dir
            prepare_transcripts_eval(tokenizer=tokenizer, max_length= args.chunk_size, save_path = predict_data_dir)
        
        # predict_data = load_data(args.predict_dir)
        if args.load_checkpoint:

            if device == torch.device('cpu'):
                checkpoint = torch.load(os.path.join(
                    model_dir, args.model_name+'best-val-acc-model.pt'), map_location=device)
            else:
                checkpoint = torch.load(os.path.join(
                    model_dir, args.model_name+'best-val-acc-model.pt'))
            classifier.load_state_dict(checkpoint['state_dict'])

        # load data
        dataset = load_dataset('./oneinamillionwrapper/transcript_evaldataset.py', download_mode="force_redownload", data_dir= predict_data_dir)
        dataset = dataset['test']
        splited_nums = dataset['split_nums']

        # predict_loader = DataLoader(BinaryDescLMDataset(data=predict_data['all_trans'], prompt=prompt, label2name=label2name, pretrained_model= args.pretrained_model, do_train=False, random_mask=False), batch_size=args.batch_size, shuffle=False)
        if args.use_prompt:
            logging.info('Prompt based method.....')
            encoded_dataset = dataset.map(lambda e: prompt_encoding(e['transcript'], tokenizer, args.prompt, args.max_length), batched=True, remove_columns=['transcript', 'codes', 'split_nums'])
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            
            # encoded_dataset = dataset.map(lambda e: prompt_encoding(e['transcript'], tokenizer, args.prompt), batched=True, remove_columns=dataset.column_names)
            predict_dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)
            id2label = None

        else:
            logging.info('conventional classifier...')
            with open(os.path.join(args.data_dir, args.label_path), 'r') as f:
                label2id = json.load(f)
            id2label = {label2id[label]: label2name[label] for label in label2id}
            logging.info('id2label',id2label)
            encoded_dataset = dataset.map(lambda e: tokenizer(e['all_segments'], padding=True, truncation=True, max_length=args.max_length), batched=True, remove_columns=dataset.column_names)
            encoded_dataset = encoded_dataset.map(lambda e: {'targets': len(e['input_ids'])*[0]}, batched=True)
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            predict_dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)

        class_names = list(label2name.values())
        predictions, _ = classifier.predict(predict_dataloader, device, tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(class_names))
        # predictions,_ = classifier.predict(predict_dataloader, device, pos_id=tokenizer.convert_tokens_to_ids('really'), neg_id = tokenizer.convert_tokens_to_ids('not'), tokenizer=tokenizer, id2class=id2label, use_prompt=args.use_prompt, class_names=np.array(list(label2name.values())))
        # plot_heatmap(class_logits, predict_data['splited_nums'], class_names)

        mult_lbl_enc = MultiLabelBinarizer()
        labels = [[label2name[code] for code in codes] for codes in dataset['codes']]
        y_hot = mult_lbl_enc.fit_transform(labels)
        predictions = mult_lbl_enc.transform(predictions)

        final_predictions = merge_predictions(splited_nums, np.array(predictions))
        
        logging.info(evaluate_classifications(y_hot, final_predictions, list(mult_lbl_enc.classes_), show_report=True))