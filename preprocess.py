import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
from nltk.tokenize import word_tokenize
import os
from collections import Counter
import random
import pickle
import numpy as np
import pandas as pd

def read_tsv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'text'])
    data['label'] = data['label'].apply(lambda x: list(map(int, x)))
    return data['text'].tolist(), data['label'].tolist()

def preprocessing(args):
    train_path = os.path.join(args.data_path, 'train.tsv')
    # test_path = os.path.join(args.data_path, 'test.tsv')
    val_path = os.path.join(args.data_path, 'validation.tsv')

    train_texts, train_labels = read_tsv(train_path)
    # test_texts, test_labels = read_tsv(test_path)
    val_texts, val_labels = read_tsv(val_path)

    #load the tag2idx
    with open(os.path.join(args.data_path, 'tag2idx.pkl'), 'rb') as f:
        tag2idx = pickle.load(f)

    #get the idx2tag
    idx2tag = {v: k for k, v in tag2idx.items()}

    label = np.array(train_labels)
    co_mat = np.dot(label.T, label)

    # Set the diagonal elements to 0
    np.fill_diagonal(co_mat, 0)

    # prompt: normalized the co_mat_copy
    normalized_co_mat = co_mat / np.sum(co_mat, axis=1)[:, np.newaxis]

    #noise rate
    # rho = 0.2

    #L
    len_list = [sum(temp) for temp in train_labels]
    Lavg = np.mean(len_list)
    L = len(train_labels[0])
    print('Lavg, L', Lavg, L)

    rho_10 = args.rho
    rho_01 = args.rho*Lavg / (L- Lavg)
    print('rho_01, rho_10', rho_01, rho_10)

    #noise of train
    fn_label = []
    for label_list in train_labels:
        fn_noise = false_negative_noise(label_list, rho_10)
        fn_label.append(fn_noise)
    print('avg noise labels num of each sample')
    print(round(np.array(fn_label).sum()/len(fn_label),2))#rho*Lavg

    fp_label = []
    for label_list in train_labels:
        fp_noise = false_positive_noise(normalized_co_mat, label_list, tag2idx, rho_01)
        fp_label.append(fp_noise)
    print('avg noise labels num of each sample')
    print(round(np.array(fp_label).sum()/len(fn_label),3))

    train_labels_with_noise = []
    for i in range(len(train_labels)):
        train_i = train_labels[i]
        fn_i = fn_label[i]
        fp_i = fp_label[i]
        noise_i = np.array(train_i) - np.array(fn_i) + np.array(fp_i)
        train_labels_with_noise.append(noise_i)
    
    #noise of validation 
    fn_label_val = []
    for label_list in val_labels:
        fn_noise = false_negative_noise(label_list, rho_10)
        fn_label_val.append(fn_noise)
    
    fp_label_val = []
    for label_list in val_labels:
        fp_noise = false_positive_noise(normalized_co_mat, label_list, tag2idx, rho_01)
        fp_label_val.append(fp_noise)

    val_labels_with_noise = []
    for i in range(len(val_labels)):
        val_i = val_labels[i]
        fn_i = fn_label_val[i]
        fp_i = fp_label_val[i]
        noise_i = np.array(val_i) - np.array(fn_i) + np.array(fp_i)
        val_labels_with_noise.append(noise_i)

    #save
    save_train_path = os.path.join(args.data_path,'train_noise_%s.tsv'%str(args.rho))
    save_val_path = os.path.join(args.data_path,'validation_noise_%s.tsv'%str(args.rho))
    print('save path', save_train_path, save_val_path)

    #save train to .tsv
    df = pd.DataFrame({'text': train_texts, 'label': train_labels_with_noise})
    # Save the DataFrame to a TSV file
    df.to_csv(save_train_path, sep='\t', index=False)    

    #save val to .tsv
    df = pd.DataFrame({'text': val_texts, 'label': val_labels_with_noise})
    # Save the DataFrame to a TSV file
    df.to_csv(save_val_path, sep='\t', index=False)

    print('---%s---'%save_train_path)
    print('---%s---'%save_val_path)
    print('Finish saving the noise data')


def false_negative_noise(label_list, rho_10):
    all_zero_label = [0]*len(label_list)
    for i, label in enumerate(label_list):
        if label == 0:
            continue
        else:
          if np.random.binomial(1, rho_10)==1:
              all_zero_label[i] = 1
    return all_zero_label

def false_positive_noise(normalized_co_mat, label_list, tag2idx, rho_01):
    #compute the nosie transation matrix
    trans_vec = np.zeros(len(label_list))
    for i, label in enumerate(label_list):
        if label == 1:
            trans_vec += normalized_co_mat[i]
        else:
          continue
    trans_vec = trans_vec/sum(label_list)*rho_01*100

    #noise label do not become clean label
    for i, label in enumerate(label_list): 
        if label == 1:
            trans_vec[i] = 0

    #get the noise by prob
    all_zero_label = [0]*len(label_list)
    for i, label in enumerate(label_list):
        if label == 1:
            continue
        else:
          prob = trans_vec[i]
          if np.random.binomial(1, prob)==1:
              all_zero_label[i] = 1
    return all_zero_label



def load_txt_data(txt_path, data_size, split_token):
    max_src_len = 0
    max_trg_len = 0
    src = []
    trg = []
    i = 0
    f=open(txt_path, 'r')
    for line in f.readlines():
        # process src and trg line 
        lineVec = line.strip().split(split_token)#split by 
        src_line = lineVec[0]
        trg_line = lineVec[1]   
        src_word_list = src_line.strip().split(' ') 
        trg_word_list = trg_line.strip().split(';') 
        if len(src_word_list)>max_src_len:
            max_src_len = len(src_word_list)
        if len(trg_word_list)>max_trg_len:
            max_trg_len = len(trg_word_list)

        src.append(src_line)
        trg.append(trg_word_list)
        i+=1
        if i>= data_size:
            break

    assert len(src) == len(trg), \
        'the number of records in source and target are not the same'
    
    print('max_src_len', max_src_len)
    print('max_trg_len', max_trg_len)
    
    print("Finish reading %d lines" % len(src))
    return src, trg


def build_vocab(tokenized_src_trg_pairs, vocab_size):
    '''
    Build the vocabulary from the training (src, trg) pairs
    :param tokenized_src_trg_pairs: list of (src, trg) pairs
    :return: word2idx, idx2word, token_freq_counter
    '''
    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    token_freq_counter = Counter()
    token_freq_counter_tag = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        token_freq_counter_tag.update(trg_word_lists)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<unk>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    tag2idx = dict()
    idx2tag = dict()

    sorted_tag2idx = sorted(token_freq_counter_tag.items(), key=lambda x: x[1], reverse=True)

    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total vocab_size: %d, predefined vocab_size: %d" % (len(word2idx), vocab_size))
    print("Total tag_size: %d" %len(tag2idx))   
    
    return word2idx, idx2word, token_freq_counter, tag2idx, idx2tag


def get_label_dictionary(trg):
    tag2idx = dict()
    idx2tag = dict()
    token_freq_counter = Counter()
    for label_list in trg:
        token_freq_counter.update(label_list)
    sorted_tag2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total tag_size: %d" %len(tag2idx))       
    return tag2idx, idx2tag

def encode_one_hot(inst, vocab_size):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value]=1
    return one_hots

def list2numpy(tag2idx, trg):
    label = []
    for idx, targets in enumerate(trg):
        label_list = [tag2idx[w] for w in targets if w in tag2idx]
        label.append(label_list)
    label =  [encode_one_hot(inst, len(tag2idx)) for inst in label] 
    label= np.array(label)
    print('label.shape', label.shape)
    return label

def normalized(co_mat):
    for i, row in enumerate(co_mat):
        co_mat[i,i] = 0
        if sum(row)<=0:
            continue
        co_mat[i] = row/sum(row)
    return co_mat

