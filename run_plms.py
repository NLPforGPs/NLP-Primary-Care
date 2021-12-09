import json
from torch.utils.data import DataLoader
from nn_model import DescClassifier
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForSequenceClassification, BertForNextSentencePrediction
from utils.metrics import evaluate_classifications, evaluate_probabilities, error_analysis
import numpy  as np
import os
import logging
import datasets
from datasets import load_dataset, set_caching_enabled
from prepare_data import generate_descriptions, prepare_transcripts_eval, generate_binary_descriptions
from utils.preprocessing.data import masking, labelmapping, prediction_encoding, cks2icpc, NSP, binary_predictiton_encoding
from utils.utils import merge_predictions, one_hot_encode
from oneinamillion.resources import PCC_BASE_DIR, ICPC2CKS, PCC_CKS_DIR, DL_DATA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--stop_epochs', type=int, default=3)
    parser.add_argument('--pretrained_model', type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)


    parser.add_argument('--train_data_dir', type=str, default='desc', help='directory for train data')
    parser.add_argument('--multi_data_path', type=str, default='multiclass/coarse_grained', help='directory for processed data in multiclass methods')
    parser.add_argument('--binary_data_path', type=str, default='binary/coarse_grained', help='directory for processed data in the binary method')
    parser.add_argument('--label_path', type=str, default='label2id.json', help='path of saved label2id mapping for conventional classifier')
    parser.add_argument('--ea_file', type=str, default='error_analysis.xls', help='file of eror analysis')
    parser.add_argument('--predict_data_dir', type=str, default='/data/transcript', help='directory for data to predict')
    parser.add_argument('--model_dir', type=str, default='./model', help='directory to save model')
    parser.add_argument('--model_name', type=str, default='test', help='name of the saved model')
    parser.add_argument('--ckpt_name', type=str, default='test', help='name of the loaded model')
    parser.add_argument('--prompt', type=str, default='This is a problem of {}.', help='prompt')
    parser.add_argument('--selected_mode', type=str, default='CKS only', help='choose to use which description [CKS only, ICPC only]')
    parser.add_argument('--chunk_size', type=int, default=490, help='used to split transcripts and descriptions into small chunks')
    parser.add_argument('--max_length', type=int, default=512, help='max length used in PLMs')
    
    # boolean 
    parser.add_argument('--use_mlm', default=False, action="store_true", help='use MLM task (donot include it unless you use it)')
    parser.add_argument('--use_nsp', default=False, action="store_true", help='use NSP task to do binary classification (donot include it unless you use it)')
    parser.add_argument('--do_train',  default=False, action="store_true", help='train a model (do not include unless you use it)')
    parser.add_argument('--do_predict',default=False, action="store_true", help='test the model (do not include unless you use it)')
    parser.add_argument('--load_checkpoint', default=False, action="store_true", help='load checkpoint for cotinual training and prediction (do not include unless you use it)')
    parser.add_argument('--fine_grained_desc', default=False, action="store_true", help='load fine-grained categories (do not include unless you use it)')
    parser.add_argument('--do_error_analysis', default=False, action="store_true", help=' genearte deatils of each example (do not include unless you use it)')


    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    

    # training data directory, descriptions
    multi_train_data_dir = os.path.join(DL_DATA, args.train_data_dir, args.selected_mode.lower().replace(' ','_'), args.multi_data_path)
    binary_train_data_dir = os.path.join(DL_DATA, args.train_data_dir, args.selected_mode.lower().replace(' ','_'), args.binary_data_path)


    # prediction data directory, transcripts
    predict_data_dir = os.path.join(DL_DATA, args.predict_data_dir)
    # model save path
    model_dir = os.path.join(PCC_BASE_DIR, args.model_dir)

    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
    # This map would be used in prompt to predict real word
    label2name = {'A':'general', 'B':'blood', 'D':'digestive','F':'eye', 'H':'ear', 'K':'cardiovascular','L':'musculoskeletal',
    'N':'neurological','P':'psychological','R':'respiratory','S':'skin','T':'endocrine', 'U':'urological','W':'pregnancy','X':'female',
    'Y':'male'}

    if args.use_mlm:
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    elif args.use_nsp:
        model = BertForNextSentencePrediction.from_pretrained(args.pretrained_model)
    else:

    #     config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2name))
        model = BertForSequenceClassification.from_pretrained(args.pretrained_model)
    model.to(device)
    classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    if args.do_train:
        logging.info('training...')
        if args.use_nsp:
            if not os.path.exists(binary_train_data_dir): # prepare dataset for model training
                logging.info(f'{binary_train_data_dir} not exists, generating data....')
                os.makedirs(binary_train_data_dir)
                generate_binary_descriptions(tokenizer=tokenizer, chunk_size = args.chunk_size, test_size=0.2, selected_mode=args.selected_mode, multiclass_desc_path=multi_train_data_dir, save_path=binary_train_data_dir, fine_grained=args.fine_grained_desc)
        else:
            if not os.path.exists(multi_train_data_dir):
                logging.info(f'{multi_train_data_dir} not exists, generating data....')
                os.makedirs(multi_train_data_dir)
                # this will generate a train and test datasets in data_dir
                generate_descriptions(tokenizer=tokenizer, chunk_size = args.chunk_size, test_size = 0.2, selected_mode=args.selected_mode, save_path = multi_train_data_dir, fine_grained = args.fine_grained_desc)
                

        # using data loader script to load data
        if args.use_nsp:
            dataset = load_dataset('./oneinamillionwrapper/binary_dataset.py', download_mode="force_redownload", data_dir= binary_train_data_dir)

        else:
            dataset = load_dataset('./oneinamillionwrapper/description_dataset.py', download_mode="force_redownload", data_dir= multi_train_data_dir)
            
        if args.use_mlm:
            # using masking method to generate examples
            logging.info('masking tokens...')
            dataset = dataset.map(lambda e: masking(e['description'], e['codes'], label2name, tokenizer, args.prompt), batched=True)

        elif args.use_nsp:
            # print('desc', len(dataset['train']['description']))
            dataset = dataset.map(lambda e: NSP(e['description'], e['codes'], e['polarity'], label2name, tokenizer, args.prompt, fine_grained=args.fine_grained_desc), batched=True)

        else:
            logging.info('using traditional bert classifier...')
            label_path = os.path.join(DL_DATA, args.label_path)
            if os.path.exists(label_path):
                logging.info('label file has existed!, using it...')
                with open(label_path,'r') as f:
                    label2id = json.load(f)
            else:
                labels = list(set(dataset['train']['codes']))
                label2id = {key: ix for ix, key in enumerate(labels)}
        
                logging.info(f'save label file to {label_path}')
                with open(os.path.join(label_path), 'w') as f:
                    json.dump(label2id, f)
            print('label2id',label2id)

             ### The reason I initialize it here is the model needs num_labels to initia lize, but for coarse-grained and fine-grained num_labels are different
            config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2id))
            model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
            model.to(device)
            classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)

            dataset = dataset.map(lambda e: labelmapping(e['codes'], label2id), batched=True)
            dataset = dataset.map(lambda e: tokenizer(e['description'], padding='max_length', truncation=True, max_length=args.max_length), batched=True)

        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
        train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False)
        
        classifier.train(train_loader=train_dataloader, dev_loader=dev_dataloader, save_dir=model_dir, save_name=args.model_name, stop_epochs=args.stop_epochs, device=device, prompt=args.prompt, load_checkpoint=args.load_checkpoint, ckpt_name=args.ckpt_name)

    if args.do_predict:
        logging.info('Predicting...')

        # if args.use_mlm:
        #     model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
        # elif args.use_nsp:
        #     model = BertForNextSentencePrediction.from_pretrained(args.pretrained_model)
        # else:
        #     config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2name))
        #     model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)

        # model.to(device)
        # classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)
        if not os.path.exists(predict_data_dir): # this will generate a train and test datasets in adata_dir
            logging.info(f'{predict_data_dir} not exists, generating data....')
            os.makedirs(predict_data_dir)
            prepare_transcripts_eval(tokenizer=tokenizer, max_length= args.chunk_size, save_path = predict_data_dir)
        
        # predict_data = load_data(args.predict_dir)

        # load data
        dataset = load_dataset('./oneinamillionwrapper/transcript_evaldataset.py', download_mode="force_redownload", data_dir= predict_data_dir)
        dataset = dataset['test']
        splited_nums = dataset['split_nums']

        # load label2id used for conventional BERT classifier and fined-grained categories.
        with open(os.path.join(DL_DATA, args.label_path), 'r') as f:
            label2id = json.load(f)
        if args.fine_grained_desc:
            id2label = {label2id[label]: label for label in label2id}
        else:
            id2label = {label2id[label]: label2name[label] for label in label2id}

        # convert labels to one-hot encoding for merging
        y_hot, mult_lbl_enc = one_hot_encode(dataset['codes'], label2name)
        class_names = list(mult_lbl_enc.classes_)

        # predict_loader = DataLoader(BinaryDescLMDataset(data=predict_data['all_trans'], prompt=prompt, label2name=label2name, pretrained_model= args.pretrained_model, do_train=False, random_mask=False), batch_size=args.batch_size, shuffle=False)
        if args.use_mlm:
            logging.info('MLM method.....')
            encoded_dataset = dataset.map(lambda e: prediction_encoding(e['transcript'], tokenizer, args.prompt, args.max_length), batched=True, remove_columns=['transcript', 'codes', 'split_nums','record_id'])
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            
            id2label = None

        elif args.use_nsp:
            logging.info('NSP method')
            if args.fine_grained_desc:
                with open(os.path.join(DL_DATA, args.label_path), 'r') as f:
                    label2id = json.load(f)
                # id2label = {label2id[label]: label for label in label2id}
                class_names = list(label2id.keys())
            encoded_dataset = dataset.map(lambda e: binary_predictiton_encoding(e['transcript'], tokenizer, args.prompt, class_names, args.max_length), batched=True, remove_columns=['transcript', 'codes', 'split_nums', 'record_id'])
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])
            id2label = None

        else:
            logging.info('conventional classifier...')
            with open(os.path.join(DL_DATA, args.label_path), 'r') as f:
                label2id = json.load(f)
            if args.fine_grained_desc:
                id2label = {label2id[label]: label for label in label2id}
            else:
                id2label = {label2id[label]: label2name[label] for label in label2id}

            config = BertConfig.from_pretrained(args.pretrained_model, num_labels=len(label2id))
            model = BertForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
            model.to(device)
            classifier = DescClassifier(model = model, epochs = args.epochs, learning_rate = args.learning_rate, weight_decay = args.weight_decay)
            logging.info(f'id to label is {id2label}')
            
            encoded_dataset = dataset.map(lambda e: prediction_encoding(e['transcript'], tokenizer, args.prompt, args.max_length, withoutmask=True), batched=True, remove_columns=dataset.column_names)
            encoded_dataset = encoded_dataset.map(lambda e: {'targets': len(e['input_ids'])*[0]}, batched=True)
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])

        predict_dataloader = DataLoader(encoded_dataset, batch_size=args.batch_size, shuffle=False)

        if not args.load_checkpoint:
            raise ValueError("You are using the pretrained models to predict, please load checkpoint at first!")

        if device == torch.device('cpu'):
            checkpoint = torch.load(os.path.join(
                model_dir, args.model_name+'best-val-acc-model.pt'), map_location=device)
        else:
            checkpoint = torch.load(os.path.join(
                model_dir, args.model_name+'best-val-acc-model.pt'))
        classifier.load_state_dict(checkpoint['state_dict'])

        predictions, pred_probs = classifier.predict(predict_dataloader, device, tokenizer=tokenizer, id2class=id2label, use_mlm=args.use_mlm, use_nsp=args.use_nsp, class_names=np.array(class_names))
        # print('raw_predictions',len(predictions))

        ## Generate error analysis file
        if args.do_error_analysis:
            ea_output_path = os.path.join(DL_DATA, args.ea_file)
            error_analysis(predictions, pred_probs, splited_nums, dataset['transcript'], dataset['record_id'], ea_output_path)
            # exit()
        ##### Evaluation Part
        if args.fine_grained_desc:
            map_file = os.path.join(PCC_CKS_DIR, ICPC2CKS)
            # map_file = os.path.join(PCC_CKS_DIR, ICPC2CKS.replace('.json','_1.json'))
            cks2icpc_dic = cks2icpc(map_file)
            mapped_predictions = []
            for pred in predictions:
                labels_per_item = []
                for label in pred:
                    labels_per_item.extend([label2name[icpc] for icpc in cks2icpc_dic[label] if icpc != 'Z'])
                mapped_predictions.append(labels_per_item)
            # print('raw', predictions)
            # print('predictions',mapped_predictions)
            predictions = mult_lbl_enc.transform(mapped_predictions)
        # print('convert predictions', len(predictions))
        # print(cks2icpc_dic)
        # print('cumsum', np.cumsum(splited_nums))
        else:
            predictions = mult_lbl_enc.transform(predictions)

        # merge labels for each transcript
        final_predictions = merge_predictions(splited_nums, np.array(predictions))
        final_probs = merge_predictions(splited_nums, np.array(pred_probs), probs=True)
        # calculate f1 score and auc_roc score
        f1_score = evaluate_classifications(y_hot, final_predictions, class_names, show_report=True)
        print('f1_score', f1_score)
        if not args.fine_grained_desc:
            auc = evaluate_probabilities(y_hot, final_probs)
            print('roc_auc', auc)