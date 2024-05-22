import torch
import random
from random import *
import torch.utils.data as Data
import pandas as pd
import numpy as np
from utility import shuffle_and_split_data

class Dataset_for_epitope(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
  
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]

class Dataset_for_beta(Data.Dataset):
    def __init__(self, input_ids, output_labels, mask_for_loss):
        self.input_ids = input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
  
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx]
    
class Dataset_for_seq2seq(Data.Dataset):
    def __init__(self, dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos):
        self.dec_input_ids = dec_input_ids
        self.output_labels = output_labels
        self.mask_for_loss = mask_for_loss
        self.enc_input_ids = enc_input_ids
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.dec_input_ids)
    
    def __getitem__(self, idx):
        return self.dec_input_ids[idx], self.output_labels[idx], self.mask_for_loss[idx], self.enc_input_ids[idx], self.masked_pos[idx]

class Dataset_for_generate(Data.Dataset):
    def __init__(self, input, mask):
        self.input = input
        self.mask = mask

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        return self.input[index], self.mask[index]


def make_data_for_pretrain(data_path, vocab_path, maxlen, max_pred):
    data_df = pd.read_csv(data_path)
    seq_list = [seq for seq in data_df['epitope']]
    token_df = pd.read_csv(vocab_path)
    token_list = [token for token in token_df['token']]
    vocab_freq_dict = {token:frequency for token, frequency in zip(token_df['token'],token_df['frequency']) }
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYW"]
    token_list = token_list+amino_acids
    token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
    for i, w in enumerate(token_list):
        token2idx[w] = i + 4
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)
    token_idx_list = []
    for seq in seq_list:
        tokens = []
        i = 0
        if len(seq)>30:
            seq = seq[:30]
        seq_len = len(seq)
        while i < seq_len:
            if i > seq_len - 2:
                tokens.append(token2idx[seq[i]])
                i += 1
            else:
                temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                temp_token_freq = [vocab_freq_dict[token] if token in vocab_freq_dict.keys() else 0 for token in temp_token_list]
                if sum(temp_token_freq) == 0:
                    tokens.append(token2idx[seq[i]])
                    i += 1
                else:
                    selected_token = temp_token_list[np.argmax(temp_token_freq)]
                    tokens.append(token2idx[selected_token])
                    i += len(selected_token)
        token_idx_list.append(tokens)
    
    batch = []
    for seq in token_idx_list:
        input_ids = [token2idx['[CLS]']] + seq + [token2idx['[SEP]']]
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != token2idx['[CLS]'] and token != token2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = token2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        batch.append([input_ids, masked_tokens, masked_pos])
    
    train_batch, validate_batch = shuffle_and_split_data(batch, ratio=0.01)
    input_ids, masked_tokens, masked_pos = zip(*train_batch)
    input_ids, masked_tokens, masked_pos = \
    torch.LongTensor(input_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos)
    input_ids_v, masked_tokens_v, masked_pos_v = zip(*validate_batch)
    input_ids_v, masked_tokens_v, masked_pos_v = \
    torch.LongTensor(input_ids_v), torch.LongTensor(masked_tokens_v),\
    torch.LongTensor(masked_pos_v)
    
    train_dataset = Dataset_for_epitope(input_ids, masked_tokens, masked_pos)
    validate_dataset = Dataset_for_epitope(input_ids_v, masked_tokens_v, masked_pos_v)


    return train_dataset, validate_dataset, vocab_size

def make_data_for_gpt_pretrain(data_path, vocab_path, max_len):
    data_df = pd.read_csv(data_path)
    seq_list = [seq for seq in data_df['beta']]
    token_df = pd.read_csv(vocab_path)
    token_list = [token for token in token_df['token']]
    vocab_freq_dict = {token:frequency for token, frequency in zip(token_df['token'],token_df['frequency']) }
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYW"]
    token_list = token_list+amino_acids
    token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2}
    for i, w in enumerate(token_list):
        token2idx[w] = i + 3
    idx2token = {i: w for i, w in enumerate(token2idx)}
    vocab_size = len(token2idx)
    token_idx_list = []
    for seq in seq_list:
        tokens = []
        i = 0
        if len(seq)>30:
            seq = seq[:30]
        seq_len = len(seq)
        while i < seq_len:
            if i > seq_len - 2:
                tokens.append(token2idx[seq[i]])
                i += 1
            else:
                temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                temp_token_freq = [vocab_freq_dict[token] if token in vocab_freq_dict.keys() else 0 for token in temp_token_list]
                if sum(temp_token_freq) == 0:
                    tokens.append(token2idx[seq[i]])
                    i += 1
                else:
                    selected_token = temp_token_list[np.argmax(temp_token_freq)]
                    tokens.append(token2idx[selected_token])
                    i += len(selected_token)
        token_idx_list.append(tokens)
    
    batch = []
    for seq in token_idx_list:
        input_ids = [token2idx['[CLS]']] + seq
        output_labels = seq + [token2idx['[SEP]']]
        mask_for_loss = [1] * len(input_ids)
        n_pad1 = max_len - len(input_ids)
        n_pad2 = max_len - len(output_labels)
        if n_pad1>0:
            input_ids.extend([0] * n_pad1)
            mask_for_loss.extend([0] * n_pad1)
        if n_pad2>0:
            output_labels.extend([0] * n_pad2)

        batch.append([input_ids, output_labels, mask_for_loss])
    train_batch, validate_batch = shuffle_and_split_data(batch, ratio=0.01)
    input_ids, output_labels, mask_for_loss = zip(*train_batch)
    input_ids, output_labels, mask_for_loss = \
    torch.LongTensor(input_ids), torch.LongTensor(output_labels),\
    torch.LongTensor(mask_for_loss)
    input_ids_v, output_labels_v, mask_for_loss_v = zip(*validate_batch)
    input_ids_v, output_labels_v, mask_for_loss_v = \
    torch.LongTensor(input_ids_v), torch.LongTensor(output_labels_v),\
    torch.LongTensor(mask_for_loss_v)
    
    train_dataset = Dataset_for_beta(input_ids, output_labels, mask_for_loss)
    validate_dataset = Dataset_for_beta(input_ids_v, output_labels_v, mask_for_loss_v)


    return train_dataset, validate_dataset, vocab_size
def make_data_for_seq2seq(data_path, tcr_vocab, pep_vocab, max_len, mode):

    data = pd.read_csv(data_path)
    tcr_seq_list = [seq for seq in data['beta']]
    pep_seq_list = [seq for seq in data['epitope']]
    amino_acids = [ac for ac in "RHKDESTNQCUGPAVILMFYW"]

    tcr_token_df = pd.read_csv(tcr_vocab)
    tcr_token_list = [token for token in tcr_token_df['token']]
    tcr_token_list = tcr_token_list+amino_acids
    tcr_token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2}
    for i, w in enumerate(tcr_token_list):
        tcr_token2idx[w] = i + 3
    tcr_idx2token = {i: w for i, w in enumerate(tcr_token2idx)}
    tcr_vocab_size = len(tcr_token2idx)
    tcr_vocab_freq_dict = {token:frequency for token, frequency in zip(tcr_token_df['token'],tcr_token_df['frequency']) }
    tcr_token_idx_list = []
    for seq in tcr_seq_list:
        tokens = []
        i = 0
        if len(seq)>30:
            seq = seq[:30]
        seq_len = len(seq)
        while i < seq_len:
            if i > seq_len - 2:
                tokens.append(tcr_token2idx[seq[i]])
                i += 1
            else:
                temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                temp_token_freq = [tcr_vocab_freq_dict[token] if token in tcr_vocab_freq_dict.keys() else 0 for token in temp_token_list]
                if sum(temp_token_freq) == 0:
                    tokens.append(tcr_token2idx[seq[i]])
                    i += 1
                else:
                    selected_token = temp_token_list[np.argmax(temp_token_freq)]
                    tokens.append(tcr_token2idx[selected_token])
                    i += len(selected_token)
        tcr_token_idx_list.append(tokens)

    pep_token_df = pd.read_csv(pep_vocab)
    pep_token_list = [token for token in pep_token_df['token']]
    pep_token_list = pep_token_list+amino_acids
    pep_token2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
    for i, w in enumerate(pep_token_list):
        pep_token2idx[w] = i + 4
    pep_idx2token = {i: w for i, w in enumerate(pep_token2idx)}
    pep_vocab_size = len(pep_token2idx)
    pep_vocab_freq_dict = {token:frequency for token, frequency in zip(pep_token_df['token'],pep_token_df['frequency']) }
    pep_token_idx_list = []
    for seq in pep_seq_list:
        tokens = []
        i = 0
        if len(seq)>30:
            seq = seq[:30]
        seq_len = len(seq)
        while i < seq_len:
            if i > seq_len - 2:
                tokens.append(pep_token2idx[seq[i]])
                i += 1
            else:
                temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                temp_token_freq = [pep_vocab_freq_dict[token] if token in pep_vocab_freq_dict.keys() else 0 for token in temp_token_list]
                if sum(temp_token_freq) == 0:
                    tokens.append(pep_token2idx[seq[i]])
                    i += 1
                else:
                    selected_token = temp_token_list[np.argmax(temp_token_freq)]
                    tokens.append(pep_token2idx[selected_token])
                    i += len(selected_token)
        pep_token_idx_list.append(tokens)

    batch = []
    for seq_tcr, seq_pep in zip(tcr_token_idx_list,pep_token_idx_list):
        masked_pos = []
        dec_input_ids = [tcr_token2idx['[CLS]']] + seq_tcr
        output_labels = seq_tcr + [tcr_token2idx['[SEP]']]
        mask_for_loss = [1] * len(dec_input_ids)
        n_pad1 = max_len - len(dec_input_ids)
        n_pad2 = max_len - len(output_labels)
        if n_pad1>0:
            dec_input_ids.extend([0] * n_pad1)
            mask_for_loss.extend([0] * n_pad1)
        if n_pad2>0:
            output_labels.extend([0] * n_pad2)
        enc_input_ids = [pep_token2idx['[CLS]']] + seq_pep + [pep_token2idx['[SEP]']]
        n_pad = max_len - len(enc_input_ids)
        enc_input_ids.extend([0] * n_pad)
        masked_pos.extend([0] * 5) #no use
        batch.append([dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos])
    train_batch, validate_batch = shuffle_and_split_data(batch, ratio=0.2)
    dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos = zip(*train_batch)
    dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos = \
    torch.LongTensor(dec_input_ids), torch.LongTensor(output_labels),\
    torch.LongTensor(mask_for_loss), torch.LongTensor(enc_input_ids), torch.LongTensor(masked_pos)

    dec_input_ids_v, output_labels_v, mask_for_loss_v, enc_input_ids_v, masked_pos_v = zip(*validate_batch)
    dec_input_ids_v, output_labels_v, mask_for_loss_v, enc_input_ids_v, masked_pos_v = \
    torch.LongTensor(dec_input_ids_v), torch.LongTensor(output_labels_v),\
    torch.LongTensor(mask_for_loss_v), torch.LongTensor(enc_input_ids_v), torch.LongTensor(masked_pos_v)

    train_dataset = Dataset_for_seq2seq(dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos)
    validate_dataset = Dataset_for_seq2seq(dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos)
    if mode == 'train':
        return train_dataset, validate_dataset, tcr_vocab_size, pep_vocab_size 
    if mode =='generate':
        pep_set = list(set(pep_seq_list))
        pep_token_idx_list = []
        for seq in pep_set:
            tokens = []
            i = 0
            if len(seq)>30:
                seq = seq[:30]
            seq_len = len(seq)
            while i < seq_len:
                if i > seq_len - 2:
                    tokens.append(pep_token2idx[seq[i]])
                    i += 1
                else:
                    temp_token_list = list(set([seq[i: i+token_len] for token_len in [2,3]]))
                    temp_token_freq = [pep_vocab_freq_dict[token] if token in pep_vocab_freq_dict.keys() else 0 for token in temp_token_list]
                    if sum(temp_token_freq) == 0:
                        tokens.append(pep_token2idx[seq[i]])
                        i += 1
                    else:
                        selected_token = temp_token_list[np.argmax(temp_token_freq)]
                        tokens.append(pep_token2idx[selected_token])
                        i += len(selected_token)
            pep_token_idx_list.append(tokens)
        batch = []
        for pep_seq in pep_token_idx_list:
            pep_input_ids = [pep_token2idx['[CLS]']] + pep_seq + [pep_token2idx['[SEP]']]
            n_pad = max_len - len(pep_input_ids)
            pep_input_ids.extend([0] * n_pad)
            batch.append([pep_input_ids,[0]*5])
        input_ids, mask = zip(*batch)
        input_ids = torch.LongTensor(input_ids)
    
        mask = torch.LongTensor(mask)
        return Dataset_for_generate(input_ids, mask), tcr_idx2token, pep_idx2token, tcr_vocab_size, pep_vocab_size


            

        





    

    
        


        



    

