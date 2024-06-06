import os
import math
import torch
import argparse
import numpy as np
from random import *
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare import make_data_for_pretrain
import warnings


warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size,len_q,len_k)

class Embedding(nn.Module):
    def __init__(self, vocab_size, maxlen=32, d_model=768):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = pos.to(x.device)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.drop(self.norm(embedding))

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.drop = nn.Dropout(0.3)
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.dense = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(0.3)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.dense(context)
        return self.norm(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=768*4, d_model=768):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ac = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        x = self.fc2(self.ac(self.fc1(x))) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.drop(x)
        x = self.norm(x+residual)
        return x

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=8):
        super(BERT, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = nn.GELU()
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, masked_pos):
        output = self.embedding(input_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return output, logits_lm

def create_parser():
    parser = argparse.ArgumentParser(description="Bert pretrain",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=True)
    parser.add_argument("--vocab_path",dest="vocab_path",type=str,help="The vocab file in .csv format.",required=True)
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to save model.", required=True)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=10, required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=32, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=64, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=5e-5, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=0, help="seed for reproductbility",required=False)
    args = parser.parse_args()
    return args

def main(args):
    accelerator = Accelerator()
    set_seed(args.random_seed)
    dataset, validate_dataset,vocab_size = make_data_for_pretrain(args.data_path, args.vocab_path, args.maxlen, max_pred=5)
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)
    device = accelerator.device
    model = BERT(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data_loader) * args.epoch
    warmup_steps = 0.1 * total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, validate_dataloader, scheduler
    )


    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for input_ids, masked_tokens, masked_pos in data_loader:
            optimizer.zero_grad()
            _, logits_lm = model(input_ids, masked_pos)
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss = loss_lm
            total_loss = total_loss + loss.item()
            accelerator.backward(loss) 
            optimizer.step()
            scheduler.step()
        accelerator.print(f"epoch:{epoch},done! loss in train_dataset:{total_loss}")
        
        model.eval()
        loss_v = 0
        with torch.no_grad():
            for input_ids, masked_tokens, masked_pos in validate_dataloader:
                _, logits_lm = model(input_ids, masked_pos)
                loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) 
                loss_lm = (loss_lm.float()).mean()
                loss = loss_lm
                loss_v = loss_v + loss.item()
            accelerator.print(f"loss in validate_dataset:{loss_v}")

    accelerator.wait_for_everyone()
   
    model = accelerator.unwrap_model(model)
    state_dict = model.state_dict()
   
    accelerator.save(state_dict, args.model_path)

            
              
if __name__=="__main__":
    args=create_parser()
    main(args)




    
    
