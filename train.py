import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from test import evaluate
from correction import label_correction
from collect import cd_collect, rel_collect, RE_cd_collect, RE_rel_collect



def train(model, optimizer, train_loader, val_loader, test_loader, args):

    num_stop_dropping = 0
    best_valid_result = 0      
    for epoch in range(args.epochs):
        if args.pretrained ==True:
            break

        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        global_step = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if global_step % args.eval_steps == 0 and global_step != 0:
                metrics = evaluate(model, val_loader, args)
                print(f'Epochs: {epoch} | Step: {global_step} | Early Stop: {num_stop_dropping} | Valid Result: {metrics}')

                valid_result = metrics['micro_f1']
                if valid_result > best_valid_result:
                    best_valid_result = valid_result
                    num_stop_dropping = 0
                    torch.save(model.state_dict(),args.model_path)            
                else:
                    num_stop_dropping += 1                    
                if num_stop_dropping >= args.early_stop_tolerance:
                    print('Have not increased for %d check points, early stop training' % num_stop_dropping)
                    break

            global_step += 1

        avg_loss = total_loss / len(train_loader)

        if epoch == args.small_loss_epoch:
            rel_overall, label_overall = rel_collect(train_loader, model, args)
            cd_overall = cd_collect(train_loader, model, args)

    # pseudo label
    if args.pretrained ==True:
        #load the model 
        model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
        print('load the pre-trained model %s ' % args.pretrained_path)

        rel_overall, label_overall = rel_collect(train_loader, model, args)
        cd_overall = cd_collect(train_loader, model, args)

    # label correction
    pseudo_overall = label_correction(rel_overall, cd_overall, label_overall, args)

    # re training
    num_stop_dropping = 0
    best_valid_result = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        global_step = 0
        for i, batch in enumerate(train_loader):
            pseudo_tensor = pseudo_overall[i*args.batch_size: (i+1)*args.batch_size]
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=pseudo_tensor)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if global_step % args.eval_steps == 0 and global_step != 0:
                metrics = evaluate(model, val_loader, args.device)
                print(f'Epochs: {epoch} | Step: {global_step} | Early Stop: {num_stop_dropping} | Valid Result: {metrics}')

                valid_result = metrics['micro_f1']
                if valid_result > best_valid_result:
                    best_valid_result = valid_result
                    num_stop_dropping = 0
                    torch.save(model.state_dict(),args.model_path)            
                else:
                    num_stop_dropping += 1                    
                if num_stop_dropping >= args.early_stop_tolerance:
                    print('Have not increased for %d check points, early stop training' % num_stop_dropping)
                    break

def RE_train_batch(model, batch, args):
    X, musk, p1, p2, y = batch
    X, musk, p1, p2, y = [torch.from_numpy(x) for x in (X, musk, p1, p2)], torch.from_numpy(np.asarray(y, dtype=np.int_))
    X, musk, p1, p2, y = [Variable(x).cuda() for x in (X, musk, p1, p2, y)]
    features, logits, loss = model.baseModel(X, musk, p1, p2, y)		
    return features, logits, loss

def train_riedel(model, train_loader, optimizer, args):

    for epoch in range(args.epochs):
        if args.pretrained ==True:
            break
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        global_step = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            features, logits, loss = RE_train_batch(model, batch, args)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epochs: {epoch} | Loss: {total_loss}')
        torch.save(model.state_dict(),args.model_path)            

        if epoch == args.small_loss_epoch:
            rel_overall, label_overall = rel_collect(train_loader, model, args)
            cd_overall = cd_collect(train_loader, model, args)

    # pseudo label
    if args.pretrained ==True:
        #load the model 
        model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
        print('load the pre-trained model %s ' % args.pretrained_path)

        rel_overall, label_overall = RE_rel_collect(train_loader, model, args)
        cd_overall = RE_cd_collect(train_loader, model, args)

    # label correction
    pseudo_overall = label_correction(rel_overall, cd_overall, label_overall, args)

    # re training
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        for i, batch in enumerate(train_loader):
            pseudo_tensor = pseudo_overall[i*args.batch_size: (i+1)*args.batch_size]
            optimizer.zero_grad()
            X, musk, p1, p2, y = batch
            features, logits, loss = model.baseModel(X, musk, p1, p2, pseudo_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epochs: {epoch} | Loss: {total_loss}')
        torch.save(model.state_dict(),args.model_path)   







