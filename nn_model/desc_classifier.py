import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, AdamW
import utils
import numpy as np
from tqdm import trange
import os
import logging
from utils.utils import save_checkpoint, softmax
from tqdm import tqdm

class DescClassifier(nn.Module):
    def __init__(self, model, epochs, learning_rate, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = learning_rate
        self.optimizer = AdamW(self.model.parameters(), self.lr, weight_decay=weight_decay)
        
    def train(self, train_loader, dev_loader, save_dir, save_name, stop_epochs, device, prompt = None, use_schduler=True, load_checkpoint=False, ckpt_name=None):
        start_epoch, best_acc = 0, 0
        if use_schduler:
            scheduler = get_linear_schedule_with_warmup(self.optimizer, len(train_loader)*self.epochs*0.1, num_training_steps = self.epochs*len(train_loader))
        if load_checkpoint:
            logging.info('loading checkpoint....')
            checkpoint = torch.load(os.path.join(
                save_dir, ckpt_name+'_best-val-acc-model.pt'))

            scheduler.load_state_dict(checkpoint['scheduler'])
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']+1

        save_epoch = -1
        for epoch in trange(start_epoch, self.epochs):
            total_loss, total_acc = 0, 0
            self.model.train()
            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device)
    
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=targets)
                loss, logits = output[0], output[1]
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if use_schduler:
                    scheduler.step()
                total_loss += loss.item()
                # accuracy
                pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                
                targets = targets.detach().cpu().numpy()
                real_targets = targets[targets!=-100]
                # print('targets', targets)
                pred = pred[targets !=-100]
                # print('pred', pred)
                total_acc += np.mean(real_targets == pred)
                
                # target = np.array(list(map(lambda x: label_map[x], target)))
                # correct += np.sum(pred == targets)
            logging.info(f'Epochs: {epoch}/{self.epochs}, train_loss: {total_loss/len(train_loader)}, train_accuracy:{total_acc/len(train_loader)}')
            dev_loss, dev_acc = self.evaluate(dev_loader, device)
            if dev_acc > best_acc:
                best_acc = dev_acc
                save_epoch = epoch
                save_checkpoint(save_dir, epoch=epoch, name=save_name+'_best-val-acc-model',
                                state_dict=self.state_dict(), optimizer=self.optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), prompt=prompt)
                logging.info(
                    f'the dev_acc is {dev_acc}, the dev_loss is {dev_loss}, save best model to {os.path.join(save_dir, save_name+"best-val-acc-model")}')
            if epoch - save_epoch > stop_epochs:
                logging.info(
                    f'stopping without any improvement after {stop_epochs}')
                break

    def evaluate(self, dev_loader, device):
        total_loss, total_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels = targets)
                loss, logits = output[0], output[1]
                # loss = self.loss_fn(output, targets)
                total_loss += loss.item()
                # accuracy
                pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                targets = targets.detach().cpu().numpy()
                # target = np.array(list(map(lambda x: label_map[x], target)))
                real_targets = targets[targets!=-100]
                pred = pred[targets !=-100]
                total_acc += np.mean(real_targets == pred)

        return total_loss/len(dev_loader), total_acc/len(dev_loader)

    def predict(self, predict_loader, device, tokenizer, class_names, use_mlm=True, use_nsp=True, id2class=None):
        
        predictions, all_logits, labels = [], [], []
        self.model.eval()
        with torch.no_grad():
            
            for batch in tqdm(predict_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device)

                input_ids = input_ids.view(-1, 512)
                attention_mask = attention_mask.view(-1, 512)
                token_type_ids = token_type_ids.view(-1, 512)
                
                if use_mlm:  # for prompt targets is natural languages ranther than labels
                    targets = targets.view(-1, 512)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels = targets)
                loss, logits = output[0], output[1]
                # logits prompt: [Batch size, sequence_length, vocab_size], conventional: [Batch size, class_nums]
                logits = logits.detach().cpu().numpy()
                if use_mlm:
                    # index of [MASK] [[1,20],[2,32],[3,20],..]
                    index = np.argwhere(targets.detach().cpu().numpy() != -100)
                    # format it into [x_aixs,y_axis]
                    index = list(zip(*index))

                    # [batch_size, vocab_size]
                    logits = logits[index]

                all_logits.extend(logits.tolist())

            if use_mlm:
                labels, class_probs = self.obtain_max_class(tokenizer, class_names, all_logits)

            elif use_nsp:
                all_logits = np.array(all_logits)
                pred_ids = np.argmax(all_logits, axis=-1)
                class_probs = []
                labels = []
                for i in range(0,len(pred_ids), len(class_names)):
                    indices = np.argwhere(pred_ids[i:i+len(class_names)]==1).flatten()
                    # x,y = list(zip(*indices))
                    labels.append(class_names[indices].tolist())
                    class_probs.append(all_logits[i:i+len(class_names), 1])
                class_probs = np.array(class_probs).reshape(-1, len(class_names))
                class_probs = softmax(class_probs)

            else:
                pred_ids = np.argmax(all_logits, axis=-1)
                if id2class is not None:
                    labels = [[id2class[item]] for item in pred_ids]
                else:
                    labels = pred_ids
                class_probs = softmax(np.array(all_logits))
                # print('labels', labels)
            return labels, class_probs


    def obtain_max_class(self, tokenizer, class_names, logits, threshold=0):
        ''' take out the max class and its probability
        logtis:[batch_size, vocab_size]
        class_names: np.array, class names
        '''        
        class_ids = tokenizer.convert_tokens_to_ids(class_names)
        # [batch_size, vocab_size]
        logits = np.array(logits)
        logits = logits[:, class_ids]
        probs = softmax(logits)
        pred_probs = np.max(probs, axis=-1)
        selected_ids = np.argmax(logits, axis=-1)
        labels = class_names[selected_ids]
        return [[labels[i]] if pred_probs[i]>=threshold else [] for i in range(len(labels))], probs