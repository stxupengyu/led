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
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            batch_rank = torch.argsort(logits, descending=True) + torch.ones_like(logits)
            batch_w = torch.min(torch.log(batch_rank) + 1, torch.tensor(args.theta).to(args.device))
            batch_rel = batch_w * loss.detach()
            rel_overall.extend(batch_rel.cpu().numpy())
            label_overall.extend(labels.cpu().numpy())

    return np.array(rel_overall), np.array(label_overall)

def cd_collect(train_loader, model, args):
    feature_dict = {idx: None for idx in range(args.label_size)}
    logit_dict = {idx: None for idx in range(args.label_size)}
    prototype_dict = {idx: None for idx in range(args.label_size)}

    with torch.no_grad():
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            features, logits, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            dict_append_batch(features, logits, labels, feature_dict, logit_dict, args)

    for key in feature_dict.keys():
        feature_list = feature_dict[key]
        logit_list = logit_dict[key]
        if logit_list is None:
            logit_list = [np.random.rand()]
        logit_threshold = get_threshold(logit_list)
        prototype_dict[key] = get_prototype(feature_list, logit_list, logit_threshold)

    cd_overall = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            features, logits, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cd = cd_cosine_dis(features, prototype_dict, args)
            cd_overall.extend(cd)

    return np.array(cd_overall)

def cd_cosine_dis(representation, prototype_dict, args):
    batch_size, label_size, hidden_size = representation.shape
    cd_batch = []
    for batch_idx in range(batch_size):
        cd_each_sample = []
        for label_idx in range(label_size):
            if prototype_dict[label_idx] is not None:
                prototype = prototype_dict[label_idx]
                vector = torch.unsqueeze(representation[batch_idx, :], 0)  # 1 * hidden_size
                cd = -1 * torch.nn.functional.cosine_similarity(vector, prototype, dim=1)
                cd_each_sample.append(cd)
        cd_batch.append(cd_each_sample)
    return np.array(cd_batch)

def dict_append_batch(representation, logits, labels, feature_dict, logit_dict, args):
    batch_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(args.label_size):
            if labels[batch_idx, label_idx] == 1:
                vector = torch.unsqueeze(representation[batch_idx, :], 0).detach().cpu().numpy()
                logit = logits[batch_idx, label_idx].detach().cpu().numpy()
                if feature_dict[label_idx] is None:
                    feature_dict[label_idx] = vector
                    logit_dict[label_idx] = logit
                else:
                    feature_dict[label_idx] = np.vstack((feature_dict[label_idx], vector))
                    logit_dict[label_idx] = np.vstack((logit_dict[label_idx], logit))

    return feature_dict, logit_dict

def get_threshold(logit_list):
    mean_logit = np.mean(logit_list)
    record = [(logit / mean_logit) if logit > mean_logit else 1 for logit in logit_list]
    return np.mean(record)

def get_prototype(feature_list, logit_list, logit_threshold):
    record = [feature for logit, feature in zip(logit_list, feature_list) if logit > logit_threshold]
    return np.mean(record, 0)

def RE_rel_collect(train_loader, model, args):
    rel_overall = []
    label_overall = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader):
            y = batch[-1]
            features, logits, loss = RE_train_batch(model, batch, args)
            batch_rank = torch.argsort(logits, descending=True) + torch.ones_like(logits)
            batch_w = torch.min(torch.log(batch_rank) + 1, torch.tensor(args.theta).to(args.device))
            batch_rel = batch_w * loss.detach()
            rel_overall.extend(batch_rel.cpu().numpy())
            label_overall.extend(y.cpu().numpy())

    return np.array(rel_overall), np.array(label_overall)

def RE_cd_collect(train_loader, model, args):
    feature_dict = {idx: None for idx in range(args.label_size)}
    logit_dict = {idx: None for idx in range(args.label_size)}
    prototype_dict = {idx: None for idx in range(args.label_size)}

    with torch.no_grad():
        for batch in tqdm(train_loader):
            features, logits, _ = RE_train_batch(model, batch, args)
            dict_append_batch(features, logits, batch[-1], feature_dict, logit_dict, args)

    for key in feature_dict.keys():
        feature_list = feature_dict[key]
        logit_list = logit_dict[key]
        if logit_list is None:
            logit_list = [np.random.rand()]
        logit_threshold = get_threshold(logit_list)
        prototype_dict[key] = get_prototype(feature_list, logit_list, logit_threshold)

    cd_overall = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            features, logits, _ = RE_train_batch(model, batch, args)
            cd = cd_cosine_dis(features, prototype_dict, args)
            cd_overall.extend(cd)

    return np.array(cd_overall)

def RE_train_batch(model, batch, args):
    X, musk, p1, p2, y = batch
    X, musk, p1, p2, y = [Variable(torch.from_numpy(x).cuda()) for x in X], [Variable(torch.from_numpy(x).cuda()) for x in musk], [Variable(torch.from_numpy(x).cuda()) for x in p1], [Variable(torch.from_numpy(x).cuda()) for x in p2], Variable(torch.from_numpy(np.asarray(y, dtype=np.int_)).cuda())
    return model.baseModel(X, musk, p1, p2, y)
