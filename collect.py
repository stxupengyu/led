import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable



def rel_collect(train_loader, model, args):
    rel_overall = []
    label_overall = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            model.to(args.device)
            model.eval()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            batch_rank = torch.argsort(logits, descending= True) + torch.ones_like(logits)
            batch_w = torch.min(torch.log(batch_rank) + 1, torch.tensor(args.theta))
            batch_rel = batch_w * loss.detach()
            batch_rel = batch_rel.cpu().numpy()
            label_overall.extend(labels.cpu().numpy())
            rel_overall.extend(batch_rel)

    rel_overall = np.array(rel_overall)
    label_overall = np.array(label_overall)
    return rel_overall, label_overall


def cd_collect(train_loader, model, args):

    feature_dict = {}
    logit_dict = {}
    prototype_dict = {}
    for idx in range(args.label_size):
        feature_dict[idx] = None
        logit_dict[idx] = None
        prototype_dict[idx] = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)            
            dict_append_batch(features, logits, labels, feature_dict, logit_dict, args)

    for key, value in feature_dict.items():
        feature_list = feature_dict[key]
        logit_list = logit_dict[key]
        if logit_list == None:
            logit_list = [np.random.rand()]
        logit_threshold = get_threshold(logit_list)
        prototype_dict[key] = get_prototype(feature_list, logit_list, logit_threshold)

    cd_overall = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)  
            cd = cd_cosine_dis(features, prototype_dict, args)
            cd_overall.extend(cd)

    cd_overall = np.array(cd_overall)
    return cd_overall

def cd_cosine_dis(representation, prototype_dict, args):
    batch_size, label_size, hidden_size = representation.shape
    cd_batch = []
    for batch_idx in range(batch_size):
        cd_each_sample = []
        for label_idx in range(label_size):
            if prototype_dict[label_idx] is not None:
                prototype = prototype_dict[label_idx]
                vector = torch.unsqueeze(representation[batch_idx, :], 0)  # 1*hidden_size
                cd = -1*torch.nn.functional.cosine_similarity(vector, prototype, dim=1)
                cd_each_sample.append(cd)
        cd_batch.append(cd_each_sample)
    cd_batch = np.array(cd_batch)
    return cd_batch

def dict_append_batch(representation, trg, logits, feature_dict, logit_dict, args):
    batch_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(args.label_size):
            if trg[batch_idx, label_idx] ==1:
                vector = torch.unsqueeze(representation[batch_idx,:], 0) #1*hidden_size
                vector = vector.detach().cpu().numpy()
                logits = logits[batch_idx, label_idx].detach().cpu().numpy()
                if feature_dict[label_idx] is None:
                    feature_dict[label_idx] = vector
                    logit_dict[label_idx] = logits
                else:
                    feature_dict[label_idx] = np.vstack((feature_dict[label_idx], vector)) 
                    logit_dict[label_idx] = np.vstack((logit_dict[label_idx], logits))

    return feature_dict, logit_dict        

def get_threshold(logit_list):
    # print(logit_list)
    mean_logit = np.mean(logit_list)
    record = []
    for logit in logit_list:
        if logit > mean_logit:
            record.append(logit/mean_logit)
        else:
            record.append(1)
    threshold = np.mean(record)
    return threshold

def get_prototype(feature_list, logit_list, logit_threshold):
    record = []
    for logit, feature in zip(logit_list, feature_list):
        if logit > logit_threshold:
            record.append(feature)
    prototype = np.mean(record, 0)
    return prototype

def RE_rel_collect(train_loader, model, args):
    rel_overall = []
    label_overall = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            model.to(args.device)
            model.eval()
            y = batch[-1]
            features, logits, loss = RE_train_batch(model, batch, args)
            batch_rank = torch.argsort(logits, descending= True) + torch.ones_like(logits)
            batch_w = torch.min(torch.log(batch_rank) + 1, torch.tensor(args.theta))
            batch_rel = batch_w * loss.detach()
            batch_rel = batch_rel.cpu().numpy()
            label_overall.extend(y.cpu().numpy())
            rel_overall.extend(batch_rel)

    rel_overall = np.array(rel_overall)
    label_overall = np.array(label_overall)
    return rel_overall, label_overall


def RE_cd_collect(train_loader, model, args):

    feature_dict = {}
    logit_dict = {}
    prototype_dict = {}
    for idx in range(args.label_size):
        feature_dict[idx] = None
        logit_dict[idx] = None
        prototype_dict[idx] = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            X, musk, p1, p2, y = batch
            features, logits, loss =  RE_train_batch(model, batch, args)    
            dict_append_batch(features, logits, y, feature_dict, logit_dict, args)

    for key, value in feature_dict.items():
        feature_list = feature_dict[key]
        logit_list = logit_dict[key]
        if logit_list == None:
            logit_list = [np.random.rand()]
        logit_threshold = get_threshold(logit_list)
        prototype_dict[key] = get_prototype(feature_list, logit_list, logit_threshold)

    cd_overall = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader), 0):
            features, logits, loss = RE_train_batch(model, batch, args)
            cd = cd_cosine_dis(features, prototype_dict, args)
            cd_overall.extend(cd)

    cd_overall = np.array(cd_overall)
    return cd_overall

def RE_train_batch(model, batch, args):
    X, musk, p1, p2, y = batch
    X, musk, p1, p2, y = [torch.from_numpy(x) for x in X], [torch.from_numpy(x) for x in musk], [torch.from_numpy(x) for x in p1], [torch.from_numpy(x) for x in p2], torch.from_numpy(np.asarray(y, dtype=np.int_))
    X, musk, p1, p2, y = [Variable(x) for x in X], [Variable(x) for x in musk], [Variable(x) for x in p1], [Variable(x) for x in p2], Variable(y)
    X, musk, p1, p2, y= [x.cuda() for x in X], [x.cuda() for x in musk], [x.cuda() for x in p1], [x.cuda() for x in p2], y.cuda()
    features, logits, loss = model.baseModel(X, musk, p1, p2, y)		
    return features, logits, loss