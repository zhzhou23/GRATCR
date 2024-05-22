import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from BERT import set_seed
import argparse
from accelerate import Accelerator
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare import make_data_for_gpt_pretrain


def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)] #seq:b*tgt_len
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(seq.device)
    return subsequence_mask

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, maxlen=32, d_model=768):
        super().__init__()
        self.pos_embedding = nn.Embedding(maxlen, d_model)  
        self.token_embedding = nn.Embedding(vocab_size, d_model)  
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos = pos.to(x.device)
        embedding = self.token_embedding(x) + self.pos_embedding(pos)
        return self.drop(self.norm(embedding))


class Feed_Forward(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.linear2(self.relu(self.linear1(x)))
        x = self.drop(x)
        x = self.layer_norm(x+residual)
        return x

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


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.feed_forward = Feed_Forward()

    def forward(self, input, mask):
        x = self.attention_block1(input, input, input, mask)
        output = self.feed_forward(x)
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, layer_num=8):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size)
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(layer_num)])

    def forward(self, input):
        dec_self_attn_pad_mask = get_attn_pad_mask(input, input)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(input)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        emb = self.embedding(input)
        for layer in self.layers:
            emb = layer(emb, dec_self_attn_mask)
        return emb


class GPT_Model(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.decoder = Decoder(vocab_size)
        self.cls = nn.Linear(d_model, vocab_size)

    def forward(self, input):
        decoder_out = self.decoder(input)
        pre = self.cls(decoder_out)
        return decoder_out, pre
def create_parser():
    parser = argparse.ArgumentParser(description="GPT pretrain",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    dataset, validate_dataset, vocab_size = make_data_for_gpt_pretrain(args.data_path, args.vocab_path, args.maxlen)
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    device = accelerator.device
    model = GPT_Model(vocab_size)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data_loader) * args.epoch
    warmup_steps = 0.1 * total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, validate_dataloader, scheduler
    )



    for epoch in range(args.epoch):
        model.train()
        loss = 0
        for input_ids, output_labels, mask_for_loss in data_loader:
            optimizer.zero_grad()
            _, logits_lm = model(input_ids)
            loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
            loss_lm = loss_lm * mask_for_loss
            loss_lm = (loss_lm.float()).mean()
            loss = loss + loss_lm.item()
            accelerator.backward(loss_lm)
            optimizer.step()
            scheduler.step()
        accelerator.print(f"epoch:{epoch},done! loss:{loss}")

        model.eval()
        loss_v = 0
        with torch.no_grad():
            for input_ids, output_labels, mask_for_loss in validate_dataloader:
                _, logits_lm = model(input_ids)
                loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
                loss_lm = loss_lm * mask_for_loss
                loss_lm = (loss_lm.float()).mean()
                loss_v = loss_v + loss_lm.item()
            accelerator.print(f"loss in validate_dataset:{loss_v}")

    
    accelerator.wait_for_everyone()
   
    model = accelerator.unwrap_model(model)
    state_dict = model.state_dict()
   
    accelerator.save(state_dict, args.model_path)
            
              
if __name__=="__main__":
    args=create_parser()
    main(args)

