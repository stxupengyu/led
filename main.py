import os
import logging
import argparse
import time
import random
import numpy as np
import datetime
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from transformers import BertTokenizer, AdamW
from tqdm import tqdm

import dataset
import train
import test
import preprocess
from model import MltcNet, RE
from utils import time_since, cprint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    #noise generation  
    parser.add_argument('--rho', type=float, 
                        help='noise rate', default=0.2)
    parser.add_argument('--preprocess', type=bool, default=True,   #True False
                        help="pre-process")

    #hyper-parameters  of led
    parser.add_argument('--alpha', type=float, 
                        help='alpha', default=0.7)
    parser.add_argument('--theta', type=float, 
                        help='theta', default=3)    
    parser.add_argument('--epsilon', type=float, 
                        help='epsilon', default=0.01)    
    parser.add_argument('--small_loss_epoch', type=int, 
                        help='small_loss_epoch', default=2)

    #dataset
    parser.add_argument("--dataset", default="aapd", type=str, #riedel rcv aapd movie
                        help="The input data directory")   
    parser.add_argument("--data_dir", default="/data/pengyu/NMLL", type=str,
                        help="The input data directory")
    parser.add_argument("--code_dir", default="/home/pengyu/code/nmll/led2024", type=str,
                        help="The input data directory")    
    parser.add_argument("--output_name", default="output.txt", type=str,
                        help="The input data directory")
    
    #bert
    parser.add_argument('--bert_name', type=str, required=False, default='bert-base')
    parser.add_argument("--bert_path", default="/data/pengyu/bert-base-uncased", type=str,
                        help="")
    parser.add_argument("--roberta_path", default="/data/pengyu/roberta-base", type=str,
                        help="")
    parser.add_argument("--xlnet_path", default="/data/pengyu/xlnet-base-cased", type=str,#xlnet-base-cased
                        help="")
    parser.add_argument('--apex', type=bool, default=False,  # True False
                        help="")
    parser.add_argument('--feature_layers', type=int, default=1,
                        help="feature layers of bert")
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="eval steps of bert")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size of bert")

    #training
    parser.add_argument('--gpuid', type=int, default=7,
                        help="gpu id")
    parser.add_argument('--epochs', type=int, default=10,
                        help="epoch of LeD")
    parser.add_argument('--early_stop_tolerance', type=int, default=30,
                        help="early stop of LeD")
    parser.add_argument('--swa_warmup', type=int, default=10,
                        help="begin epoch of swa")
    parser.add_argument('--swa_mode', type=bool, default=False,
                        help="use swa strategy")
    parser.add_argument('--gradient_clip_value', type=int, default=5.0,
                        help="gradient clip")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #riedel 
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15K_D/')
    parser.add_argument('--mode', dest='mode', type=str, help="Data mode: multi_label or multi_class", default='multi_label')
    parser.add_argument('--run_mode', dest='run_mode', type=str, help="train, test", default='train')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.1)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--pos_dim", dest='pos_dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--hdim", dest='hdim', type=int, help="Hidden layer dimension", default=100)
    parser.add_argument("--max_bags", dest="max_bags", type=int, help="Max sentences in a bag", default=5)
    parser.add_argument("--max_len", dest="max_len", type=int, help="Max description length", default=30)
    parser.add_argument('--encoder', dest='encoder', type=str, help="Encoder type", default='PCNN')
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=10)
    parser.add_argument("--test_batch", dest='test_batch', type=int, help="Test batch size", default=100)
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Saved model path", default='./')
    parser.add_argument("--resume", dest='resume', type=int, help="Resume from epoch model file", default=-1)
    parser.add_argument('--reduce_method', dest='reduce_method', type=str, help="The bag reduce methods", default='attention')
    parser.add_argument('--position', dest='position', action="store_true", help="If using position embedding", default=False)
    parser.add_argument("--eval_start", dest='eval_start', type=int, help="Epoch when evaluation start", default=90)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluation per x iteration", default=1)
    parser.add_argument("--save_m", dest='save_m', type=int, help="Number of saved models", default=1)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
    parser.add_argument('--reg_weight', dest='reg_weight', type=float, help="The regularization weight of autoencoder", default=1e-5)
    parser.add_argument('--clip', dest='clip', type=float, help="The gradient clip max norm", default=0.0)
    parser.add_argument('--init1', dest='init1', type=float, help="Initialization of p(z=1|y=1)", default=1.)
    parser.add_argument('--init2', dest='init2', type=float, help="Initialization of p(z=1|y=0)", default=0.)
    parser.add_argument('--na_init1', dest='na_init1', type=float, help="Initialization of p(z=1|y=1) for NA relation", default=1.)
    parser.add_argument('--na_init2', dest='na_init2', type=float, help="Initialization of p(z=1|y=0) for NA relation", default=0.)
    parser.add_argument('--em', dest='em', action="store_true", help="Using EM", default=False)
    parser.add_argument('--sigmoid', dest='sigmoid', action="store_true", help="Using sigmoid to activate scores", default=False)
    parser.add_argument("--per_e_step", dest='per_e_step', type=int, help="How many m steps before e step", default=500)
    parser.add_argument('--flip', dest='flip', type=float, help="Label flip rate", default=0.)
    parser.add_argument('--save_beta', type=float, help="beta", default=10000)

    #pretrained
    parser.add_argument('--pretrained', type=bool, default=False,   #True False
                        help="use pretrained LeD model")
    parser.add_argument("--pretrained_path", default='/data/pengyu/NMLL/model/MltcNet.pth', type=str,
                        help="path of pretrained LeD model")

    args = parser.parse_args()

    #paths
    args.model_dir = os.path.join(args.data_dir, 'model')
    args.data_path = os.path.join(args.data_dir, args.dataset)
    args.save_dir = os.path.join(args.code_dir, 'output')
    args.plot_dir = os.path.join(args.code_dir, 'plot')
    args.output_txt = os.path.join(args.save_dir, args.mode+'_'+args.output_name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    args.model_path = os.path.join(args.model_dir, "MltcNet_%s_%s.pth" %(args.mode, args.timemark))
    args.history_path = os.path.join(args.model_dir, "history_%s_%s.npy" %(args.mode, args.timemark))

    #log para
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in args.__dict__.items()]

    #for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #riedel
    if args.dataset == 'riedel':
        pipelint_RE(args)
    else:
        pipeline(args)

def pipeline(args):

    #Noise Generate
    if args.preprocess == True:
        preprocess.preprocessing(args)

    #Dataset
    start_time = time.time()
    logger.info('Data Loading')
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, test_loader = dataset.get_data_loader(tokenizer, args)
    load_data_time = time_since(start_time)
    logger.info('Time for loading the data: %s' %load_data_time)

    #Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    model = MltcNet(args)
    model = model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    #Training
    train.train(model, optimizer, train_loader, val_loader, test_loader, args)
    training_time = time_since(start_time)
    logger.info('Time for training: %s' %training_time)
    logger.info(f'Best Model Path: {args.model_path}')

    #Predicting
    logger.info('Predicting')
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)) 
    result = test.evaluate(model, test_loader, args)
    logger.info(f'Final Test Result: {result}')
    with open(args.output_txt, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(' |epsilon:'+args.epsilon)
        f.write(' |alpha:'+str(args.alpha))
        f.write(' |theta:'+str(args.theta))
        f.write(' |rho:'+str(args.rho))
        f.write(' |result:' +' '.join([str(i) for i in result])+'\n')
        f.close()

def pipelint_RE(args):

    # data and model
    data_loader = dataset.RE_DataLoader(args.data_dir, args.max_bags, args.max_len, mode=args.mode, flip=args.flip)
    model = RE(args.dim, args.pos_dim, args.hdim, data_loader.n_word, data_loader.n_pos, data_loader.n_relation, args.reg_weight, reduce_method=args.reduce_method, position=args.position, encode_model=args.encoder, sigmoid=args.sigmoid, init1=args.init1, init2=args.init2, na_init1=args.na_init1, na_init2=args.na_init2)	
    model = model.cuda()
    optimizer = use_optimizer(model, lr=args.lr, method=args.optimizer)
    
    # train
    dataset = dataset.RE_Dataset(data_loader, dataset='train', shuffle=True)
    trainloader = data.RE_DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
    train.train_riedel(model, trainloader, optimizer, args)

    # test
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)) 
    dataset = dataset.RE_Dataset(data_loader, dataset='test')
    testloader = dataset.Riedel_DataLoader(dataset, batch_size=args.batch, num_workers=args.n_worker, collate_fn=dataset.data_collate)
    result = test.evaluate_riedel(model, testloader, args)
    logger.info(f'Final Test Result: {result}')

    with open(args.output_txt, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(' |epsilon:'+args.epsilon)
        f.write(' |alpha:'+str(args.alpha))
        f.write(' |theta:'+str(args.theta))
        f.write(' |rho:'+str(args.rho))
        f.write(' |result:' +' '.join([str(i) for i in result])+'\n')
        f.close()


def use_optimizer(model, lr, weight_decay=0, lr_decay=0, momentum=0, rho=0.95, method='sgd'):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if method=='sgd':
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif method=='adagrad':
        return optim.Adagrad(parameters, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
    elif method=='adadelta':
        return optim.Adadelta(parameters, rho=rho, lr=lr, weight_decay=weight_decay)
    elif method=='adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif method=='rmsprop':
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)		
    else:
        raise Exception("Invalid method, option('sgd', 'adagrad', 'adadelta', 'adam')")

def resume(step, args):
    load_file = os.path.join(args.save_dir, f'{step}_step.mod.tar')
    if os.path.isfile(load_file):	
        checkpoint = torch.load(load_file)	
        return checkpoint	
    return None

if __name__ == '__main__':
    main()
