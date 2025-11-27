import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, RobertaModel, XLNetModel, BertModel
import logging
import pandas as pd
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLabelTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def read_tsv(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'text'])
    data['label'] = data['label'].apply(lambda x: list(map(int, x)))
    return data['text'].tolist(), data['label'].tolist()

def tokenize_data(texts, labels, args):
    tokenizer = get_tokenizer(args)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return encodings, labels

def get_tokenizer(args):
    if 'roberta' in args.bert_name:
        print('load roberta-base tokenizer')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    elif 'xlnet' in args.bert_name:
        print('load xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        print('load bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    return tokenizer

def get_data_loader(tokenizer, args):

    if args.rho == 0:
        train_path = os.path.join(args.data_path,'train.tsv')
        test_path = os.path.join(args.data_path, 'test.tsv')
        val_path = os.path.join(args.data_path,'validation.tsv')
    else: 
        train_path = os.path.join(args.data_path,'train_noise_%s.tsv'%str(args.rho))
        test_path = os.path.join(args.data_path, 'test.tsv')
        val_path = os.path.join(args.data_path,'validation_noise_%s.tsv'%str(args.rho))

    train_texts, train_labels = read_tsv(train_path)
    test_texts, test_labels = read_tsv(test_path)
    val_texts, val_labels = read_tsv(val_path)

    args.label_size = len(train_labels[0])
    logger.info(F'Size of Training Set: {len(train_texts)}')
    logger.info(F'Size of Validation Set: {len(val_texts)}')
    logger.info(F'Size of Test Set: {len(test_texts)}')

    train_encodings, train_labels = tokenize_data(train_texts, train_labels, args)
    test_encodings, test_labels = tokenize_data(test_texts, test_labels, args)
    val_encodings, val_labels = tokenize_data(val_texts, val_labels, args)

    train_dataset = MultiLabelTextDataset(train_encodings, train_labels)
    test_dataset = MultiLabelTextDataset(test_encodings, test_labels)
    val_dataset = MultiLabelTextDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader

class RE_DataLoader():

	def __init__(self, data_dir, max_bags=200, max_s_len=120, mode='multi_label', flip=0.0):
		self.data_dir = data_dir
		self.max_bags = max_bags
		self.max_s_len = max_s_len	
		self.mode = mode
		self.flip = flip	

		def load_dict(file):
			with open(os.path.join(data_dir, file), 'r') as f:
				n_dict = len(f.readlines())			
			with open(os.path.join(data_dir, file), 'r') as f:
				dict2id = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
				id2dict = {v: k for k, v in dict2id.items()}
			return n_dict, dict2id, id2dict	

		self.n_entity, self.entity2id, self.id2entity = load_dict('entity2id.txt')	

		self.n_relation, self.relation2id, self.id2relation = load_dict('relation2id.txt')	

		self.n_word, self.word2id, self.id2word = load_dict('word2id.txt')

		self.max_pos = 100
		self.n_pos = self.max_pos*2+1
		train_bags, train_musk, train_pos1, train_pos2, train_labels = self.get_bags(self.create_bag('train.txt', mode))

		if flip!=0:	
			flip_count = 0
			if not os.path.exists(os.path.join(data_dir, 'flip_%f.npz'%flip)):
				origin_labels = train_labels[:]
				flip_labels = []
				for i in range(len(train_labels)):
					origin = set(list(train_labels[i]))
					fliped = []
					for j in range(self.n_relation):
						if j not in origin:
							if np.random.binomial(1, flip)==1:
								flip_count += 1
								fliped.append(j)
						else:
							if np.random.binomial(1, flip)==0:
								flip_count += 1
								fliped.append(j)
					flip_labels.append(np.asarray(fliped, dtype=np.int_))
				train_labels = flip_labels
				np.savez(os.path.join(data_dir, 'flip_%f'%flip), origin_labels=origin_labels, flip_labels=flip_labels, flip_count=flip_count)
			else:
				flip_load = np.load(os.path.join(data_dir, 'flip_%f.npz'%flip))
				train_labels = flip_load['flip_labels']
				flip_count = flip_load['flip_count']


		self.n_train = len(train_bags)		
		index_list = range(self.n_train)
		random.Random(111).shuffle(index_list)
		train_select = index_list

		self.train_bags, self.train_musk, self.train_pos1, self.train_pos2, self.train_labels = [train_bags[x] for x in train_select], [train_musk[x] for x in train_select], [train_pos1[x] for x in train_select], [train_pos2[x] for x in train_select], [train_labels[x] for x in train_select]
		self.test_manual_bags, self.test_manual_musk, self.test_manual_pos1, self.test_manual_pos2, self.test_manual_labels = self.get_bags(self.create_manual_bag('manualTest.txt'))					

	def get_word2count(self):
		worddict = dict()
		with open(self.data_dir+'train.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				words = line[5].split(' ')
				for word in words:
					if word not in worddict:
						worddict[word] = 0
					worddict[word] += 1
		with open(self.data_dir+'test.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				words = line[5].split(' ')
				for word in words:
					if word not in worddict:
						worddict[word] = 0
					worddict[word] += 1		
		sort_word = sorted(worddict.items(), key=lambda x:x[1], reverse=True)
		with open(self.data_dir+'wordcount.txt', 'ab') as f:
			for item in sort_word:
				f.write('	'.join([item[0], str(item[1])])+'\n')

	def read_pre_train_embedding(self):
		vec = {}
		with open(self.data_dir+'vec.txt') as f:
			f.readline()
			for line_ in f.readlines():
				line = line_.strip().split(' ')
				if line[0]!=' ':
					vec[line[0]] = np.asarray([float(x) for x in line[1:]], dtype=np.float32)
		self.pre_w = vec

	def get_word2id(self, low_freq, high_freq):
		index = 0
		embed = []
		w2id = open(self.data_dir+'word2id.txt', 'ab')
		with open(self.data_dir+'wordcount.txt') as f:
			for line_ in f:
				line = line_.strip().split('	')
				if int(line[1])>=low_freq and int(line[1])<=high_freq:				
					if line[0] in self.pre_w:
						embed.append(self.pre_w[line[0]])
						w2id.write('	'.join([line[0], str(index)])+'\n')
						index += 1
		embed.append(np.random.rand(50))
		embed.append(np.zeros(50))
		np.save(self.data_dir+'word_embed.npy', np.asarray(embed, dtype=np.float32))

	def pos_embed(self, x):
		return max(0, min(x + self.max_pos, self.n_pos))

	def create_bag(self, datafile, mode='multi_class'):
		name_dict = dict()
		bags = []
		bag_index = 0
		with open(self.data_dir+datafile) as f:
			for line_ in f:
				line = line_.strip().split('	')	
				e1_id = line[0]
				e2_id = line[1]
				e1_name = line[2]
				e2_name = line[3]
				rel = line[4]
				sent = line[5].split(' ')[:-1]
				if mode=='multi_class':
					bag_name = '	'.join([e1_id, e2_id, rel])
				elif mode=='multi_label':
					bag_name = '	'.join([e1_id, e2_id])					
				s = []
				pos1 = 0
				pos2 = 0
				index = 0
				for word in sent:
					if word == e1_name:
						pos1 = index
					if word == e2_name:
						pos2 = index
					if word in self.word2id:
						s.append(self.word2id[word])
					else:
						s.append(self.n_word)
					index += 1
				if bag_name not in name_dict:
					name_dict[bag_name] = bag_index
					bag_index += 1
					bags.append([[], set()])
				bags[name_dict[bag_name]][0].append((s, pos1, pos2))	
				bags[name_dict[bag_name]][1].add(rel)	
		return bags	

	def create_manual_bag(self, datafile):
		bags = []
		with open(self.data_dir+datafile) as f:
			while 1:
				line = f.readline()
				if line!='':
					lines = line.strip().split('	')
					e1_id = lines[0]
					e2_id = lines[1]
					e1_name = lines[2]
					e2_name = lines[3]
					bag_name = '	'.join([e1_id, e2_id])
					rels = f.readline().strip().split('	')
					new_bag = [[], set(rels)]
					num_sents = int(f.readline().strip())
					for i in range(num_sents):
						sent = f.readline().strip().split(' ')
						s = []
						pos1 = 0
						pos2 = 0
						index = 0
						for word in sent:
							if word == e1_name:
								pos1 = index
							if word == e2_name:
								pos2 = index
							if word in self.word2id:
								s.append(self.word2id[word])
							else:
								s.append(self.n_word)
							index += 1
						new_bag[0].append((s, pos1, pos2))
					bags.append(new_bag)
					f.readline()
				else:
					break
		return bags

	def get_bags(self, bags):		
		normal_bags = []
		musk_idxs = []
		pos1_bags = []
		pos2_bags = []
		bag_labels = []
		for key in range(len(bags)):			
			sents = bags[key][0]
			bag_size = len(sents)
			start = 0
			while start<bag_size:
				bs = []
				musk_bs = []
				p1s = []	
				p2s = []
				#cut big bags into small ones			
				if start+self.max_bags>=bag_size:
					ss = sents[start:]
				else:
					ss = sents[start:start+self.max_bags]
				for s in ss:
					sent = s[0]
					pos = [s[1], s[2]]
					pos.sort()
					m_bs = []
					for i in range(self.max_s_len):
						if i >= len(sent):
							m_bs.append(0)
						elif i - pos[0]<=0:
							m_bs.append(1)
						elif i - pos[1]<=0:
							m_bs.append(2)
						else:
							m_bs.append(3)
					musk_bs.append(m_bs)
					p1s.append([self.pos_embed(i - s[1]) for i in range(self.max_s_len)])
					p2s.append([self.pos_embed(i - s[2]) for i in range(self.max_s_len)])
					if len(sent)>=self.max_s_len:
						exs = sent[:self.max_s_len]
					else:
						exs = sent
						exs.extend([self.n_word+1]*(self.max_s_len-len(sent)))
					exs = np.asarray(exs, dtype=np.int32)
					bs.append(exs) 				
				normal_bags.append(np.asarray(bs, dtype=np.int_)) # Sizes of tensors must match except in dimension 0 in each example in a batch
				musk_idxs.append(np.asarray(musk_bs, dtype=np.int_))
				pos1_bags.append(np.asarray(p1s, dtype=np.int_)) 
				pos2_bags.append(np.asarray(p2s, dtype=np.int_)) 				
				labels = []
				for l in bags[key][1]:
					if l in self.relation2id:
						labels.append(self.relation2id[l])
					else:
						labels.append(self.relation2id['NA'])				
				bag_labels.append(np.asarray(labels, dtype=np.int_))
				start = start+self.max_bags
		return normal_bags, musk_idxs, pos1_bags, pos2_bags, bag_label

class RE_Dataset(Dataset):

	def __init__(self, data_loader, dataset='train', shuffle=False):

		self.data_loader = data_loader
		self.dataset = dataset
		self.shuffle = shuffle
		if dataset=='train':
			bags, musk, pos1, pos2, pos_labels = self.data_loader.train_bags, self.data_loader.train_musk, self.data_loader.train_pos1, self.data_loader.train_pos2, self.data_loader.train_labels
		elif dataset=='test':
			bags, musk, pos1, pos2, pos_labels = self.data_loader.test_manual_bags, self.data_loader.test_manual_musk, self.data_loader.test_manual_pos1, self.data_loader.test_manual_pos2, self.data_loader.test_manual_labels	
		labels = []
		for ls in pos_labels:
			label_rep = np.zeros(self.data_loader.n_relation, dtype=np.int_)
			label_rep[ls] = 1.
			labels.append(label_rep)	
		self.index = range(len(bags))
		if shuffle:
			random.shuffle(self.index)
		self.bags = [bags[x] for x in self.index]
		self.musk = [musk[x] for x in self.index]				
		self.pos1 = [pos1[x] for x in self.index]
		self.pos2 = [pos2[x] for x in self.index]
		self.labels = [labels[x] for x in self.index]

	def data_collate(self, batch):	
		X = []
		musk_idxs = []
		p1 = []
		p2 = []		
		y = []	
		i = []
		for item in batch:
			X.append(item[0])
			musk_idxs.append(item[1])
			p1.append(item[2]) 
			p2.append(item[3])						
			y.append(item[4])
			i.append(item[5])
		return [X, musk_idxs, p1, p2, y, i]					

	def __len__(self):
		return len(self.bags)

	def __getitem__(self, idx):
		X = self.bags[idx]
		musk_idxs = self.musk[idx]
		p1 = self.pos1[idx]
		p2 = self.pos2[idx]
		y = self.labels[idx]
		i = self.index[idx]
		return X, musk_idxs, p1, p2, y, i
	


# def load_txt_data2(args):
#     max_src_len = 0
#     max_trg_len = 0
#     max_pos_len = 0
#     max_neg_len = 0
#     max_ran_len = 0
#     src = []
#     trg = []
#     pos = []
#     neg = []
#     ran = []
#     i = 0
#     f=open(os.path.join(args.data_dir, args.data_name), 'r')
#     for line in f.readlines():
#         # process src and trg line 
#         lineVec = line.strip().split(args.split_token)#split by 
#         src_line = lineVec[0]
#         trg_line = lineVec[1]   
#         pos_line = lineVec[2]   
#         neg_line = lineVec[3]  
#         ran_line = lineVec[4]  
#         src_word_list = src_line.strip().split(' ') 
#         trg_word_list = trg_line.strip().split(';') 
#         pos_word_list = pos_line.strip().split(';') 
#         neg_word_list = neg_line.strip().split(';') 
#         ran_word_list = ran_line.strip().split(';') 
#         if len(src_word_list)>max_src_len:
#             max_src_len = len(src_word_list)
#         if len(trg_word_list)>max_trg_len:
#             max_trg_len = len(trg_word_list)
#         if len(pos_word_list)>max_pos_len:
#             max_pos_len = len(pos_word_list)
#         if len(neg_word_list)>max_neg_len:
#             max_neg_len = len(neg_word_list)
#         if len(ran_word_list)>max_ran_len:
#             max_ran_len = len(ran_word_list)

#         src.append(src_line)
#         trg.append(trg_word_list)
#         pos.append(pos_word_list)
#         neg.append(neg_word_list)
#         ran.append(ran_word_list)
#         i+=1
#         if i>= args.data_size:
#             break

#     assert len(src) == len(trg), \
#         'the number of records in source and target are not the same'
#     assert len(pos) == len(trg), \
#         'the number of records in source and target are not the same'
#     assert len(neg) == len(trg), \
#         'the number of records in source and target are not the same'
#     assert len(ran) == len(trg), \
#         'the number of records in source and target are not the same'
    
#     print('max_src_len', max_src_len)
#     print('max_trg_len', max_trg_len)
#     print('max_pos_len', max_pos_len)
#     print('max_neg_len', max_neg_len)
#     print('max_ran_len', max_ran_len)
    
#     print("Finish reading %d lines" % len(src))
#     return src, trg, pos, neg, ran

# def load_txt_data3(args):
#     max_src_len = 0
#     max_trg_len = 0
#     src = []
#     trg = []
#     i = 0
#     f=open(os.path.join(args.data_dir, args.test_name), 'r')
#     for line in f.readlines():
#         # process src and trg line 
#         lineVec = line.strip().split(args.split_token)#split by 
#         src_line = lineVec[0]
#         trg_line = lineVec[1]   
#         src_word_list = src_line.strip().split(' ') 
#         trg_word_list = trg_line.strip().split(';') 
#         if len(src_word_list)>max_src_len:
#             max_src_len = len(src_word_list)
#         if len(trg_word_list)>max_trg_len:
#             max_trg_len = len(trg_word_list)

#         src.append(src_line)
#         trg.append(trg_word_list)
#         i+=1
#         if i>= args.data_size:
#             break

#     assert len(src) == len(trg), \
#         'the number of records in source and target are not the same'
    
#     print('max_src_len', max_src_len)
#     print('max_trg_len', max_trg_len)
    
#     print("Finish reading %d lines" % len(src))
#     return src, trg

# def dataSplit(tokenized_pairs, random_seed):
#     random.seed(random_seed)
#     random.shuffle(tokenized_pairs)
#     data_length = len(tokenized_pairs)
#     train_length = int(data_length*.8)
#     valid_length = int(data_length*.9)
#     train, valid, test = tokenized_pairs[:train_length], tokenized_pairs[train_length:valid_length],\
#                                      tokenized_pairs[valid_length:]  
#     return train, valid, test

# def dataSplit2(tokenized_pairs, random_seed):
#     random.seed(random_seed)
#     random.shuffle(tokenized_pairs)
#     data_length = len(tokenized_pairs)
#     valid_length = int(data_length*.9)
#     train, valid = tokenized_pairs[:valid_length],\
#                                      tokenized_pairs[valid_length:]  
#     return train, valid

# def encode_one_hot(inst, vocab_size, label_from):
#     '''
#     one hot for a value x, int, x>=1
#     '''
#     one_hots = np.zeros(vocab_size, dtype=np.float32)
#     for value in inst:
#         one_hots[value-label_from]=1
#     return one_hots

# def padding(input_list, max_seq_len, word2idx):
#     padded_batch = word2idx['<pad>'] * np.ones((len(input_list), max_seq_len), dtype=int)
#     for j in range(len(input_list)):
#         current_len = len(input_list[j])
#         if current_len <= max_seq_len:
#             padded_batch[j][:current_len] = input_list[j]
#         else:
#             padded_batch[j] = input_list[j][:max_seq_len]
#     return padded_batch

# def list2numpy(src_trgs_pairs, word2idx, tag2idx, max_seq_len, vocab_size):
#     '''
#     word2id + padding + onehot
#     '''
#     text = []
#     label = []
#     for idx, (source, targets) in enumerate(src_trgs_pairs):
#         src = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size
#                else word2idx['<unk>'] for w in source]
#         trg = [tag2idx[w] for w in targets if w in tag2idx]
#         text.append(src)
#         label.append(trg)
#     text = padding(text, max_seq_len, word2idx)
#     label =  [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in label] 
#     text = np.array(text)
#     label= np.array(label)
#     print('text.shape', text.shape)
#     print('label.shape', label.shape)
#     return text, label

# def list2numpy_label(src_trgs_pairs, tag2idx):
#     pos = []
#     neg = []
#     ran = []
#     ddd = []
#     for idx, (pos_idx, neg_idx, ran_idx, ddd_idx) in enumerate(src_trgs_pairs):
#         # print(pos_idx, neg_idx, ran_idx)
#         pos.append([tag2idx[w] for w in pos_idx if w in tag2idx])
#         neg.append([tag2idx[w] for w in neg_idx if w in tag2idx])
#         ran.append([tag2idx[w] for w in ran_idx if w in tag2idx])
#         ddd.append([tag2idx[w] for w in ddd_idx if w in tag2idx])
#     pos = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in pos]
#     neg = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in neg]
#     ran = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in ran]
#     ddd = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in ddd]
#     pos = np.array(pos)
#     neg = np.array(neg)
#     ran = np.array(ran)
#     ddd = np.array(ddd)
#     return pos, neg, ran, ddd

# def list2numpy_label_solo(trgs, tag2idx):
#     pos = []
#     for idx, pos_idx in enumerate(trgs):
#         # print(pos_idx, neg_idx, ran_idx)
#         pos.append([tag2idx[w] for w in pos_idx if w in tag2idx])
#     pos = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in pos]
#     pos = np.array(pos)
#     return pos

# def generate_lack(trg, neg):
#     lack = []
#     for target, observe in zip(trg, neg):
#         temp = []
#         for word in target:
#             if word not in observe:
#                 temp.append(word)
#         lack.append(temp)
#     return lack