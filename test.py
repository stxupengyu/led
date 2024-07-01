import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, average_precision_score
from scipy import sparse
from tqdm import tqdm

def evaluate(model, val_loader, args):
    pre_K = 10
    model.to(args.device)
    model.eval()

    y_test, y_pred = None, None

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            _, logits, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

            if y_test is None:
                y_test = labels
                y_pred = logits
            else:
                y_test = torch.cat((y_test, labels), 0)
                y_pred = torch.cat((y_pred, logits), 0)

    y_pred = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0))
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    result = evaluate_metrics(y_test, y_pred, args)
    return result


def evaluate_metrics(targets, prediction, args):
    micro_f1 = f1_score(targets, prediction, average='micro')
    macro_f1 = f1_score(targets, prediction, average='macro')
    map_score = average_precision_score(targets, prediction, average='macro')

    result = {
        'micro_f1': round(micro_f1 * 100, 2),
        'macro_f1': round(macro_f1 * 100, 2),
        'map': round(map_score * 100, 2)
    }
    return result


def evaluate_riedel(model, data_loader, args):
    model.eval()
    pred_all, y_all = [], []

    for batch_idx, (X, musk, p1, p2, y, _) in enumerate(data_loader):
        X, musk, p1, p2, y = map(lambda x: torch.from_numpy(x), [X, musk, p1, p2, np.asarray(y, dtype=np.float32)])
        X, musk, p1, p2, y = map(lambda x: Variable(x, requires_grad=False).cuda(), [X, musk, p1, p2, y])

        pred = model.pred(X, musk, p1, p2).data.cpu().numpy()
        y = y.data.cpu().numpy()

        pred_all.append(pred)
        y_all.append(y)

    y_score = np.concatenate(pred_all)
    y_test = np.concatenate(y_all)
    y_pred = np.where(y_score > 0.5, 1, 0)

    result = evaluate_metrics(y_test, y_pred, args)
    return result


def evaluate_metrics_top_k(targets, prediction, args):
    def get_precision(prediction, targets, mlb, top_K):
        targets = sparse.csr_matrix(targets)
        prediction = sparse.csr_matrix(mlb.transform(prediction[:, :top_K]))
        precision = prediction.multiply(targets).sum() / (top_K * targets.shape[0])
        return round(precision * 100, 2)

    mlb = MultiLabelBinarizer(classes=range(args.label_size))
    mlb.fit(targets)

    result = [get_precision(prediction, targets, mlb, top_K) for top_K in [1, 3, 5]]
    return result

# def evaluate_test(targets, prediction, mlb, args):

#     def get_precision(prediction, targets, mlb, top_K, args):
#         targets = sparse.csr_matrix(targets)
#         prediction = mlb.transform(prediction[:, :top_K])
#         precision = prediction.multiply(targets).sum() / (top_K * targets.shape[0])
#         # precision = evaluator(targets.A, prediction.A, top_K)
#         return round(precision * 100, 2)

#     def get_ndcg(prediction, targets, mlb, top_K, args):
#         log = 1.0 / np.log2(np.arange(top_K) + 2)
#         dcg = np.zeros((targets.shape[0], 1))
#         targets = sparse.csr_matrix(targets)
#         for i in range(top_K):
#             p = mlb.transform(prediction[:, i: i + 1])
#             dcg += p.multiply(targets).sum(axis=-1) * log[i]
#         ndcg = np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top_K) - 1])
#         return round(ndcg * 100, 2)

#     result = []
#     for top_K in [1, 3, 5]:
#         precision = get_precision(prediction, targets, mlb, top_K, args)
#         result.append(precision)

#     # for top_K in [1, 3, 5]:
#     #     ndcg = get_ndcg(prediction, targets, mlb, top_K, args)
#     #     result.append(ndcg)
#     return result

# def evaluator(y_true, y_pred, top_K):
#     precision_K = []
#     for i in range(y_pred.shape[0]):
#         if np.sum(y_true[i, :])==0:
#             continue
#         top_indices = y_pred[i].argsort()[-top_K:]
#         p = np.sum(y_true[i, top_indices]) / top_K
#         precision_K.append(p)
#     precision = np.mean(np.array(precision_K))
#     return precision