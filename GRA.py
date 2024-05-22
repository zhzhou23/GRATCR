import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from BERT import BERT, set_seed
from GPT import GPT_Model, get_attn_pad_mask, get_attn_subsequence_mask
from transformers import AdamW, get_linear_schedule_with_warmup
from Data_prepare import make_data_for_seq2seq
from beam_search import BeamHypotheses, expand_inputs
import argparse
from tqdm import tqdm
from accelerate import Accelerator

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.3)
    def forward(self,Q,K,V,attn_mask):
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(64)
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context=torch.matmul(attn,V)
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(n_heads*d_v,d_model,bias=False)
        self.norm=nn.LayerNorm(self.d_model)
    def forward(self,input_Q,input_K,input_V,attn_mask):
        batch_size=input_Q.size(0)
        residual=input_Q
        Q=self.W_Q(input_Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K=self.W_K(input_K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V=self.W_V(input_V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2) #b*heads*seq*d
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)
        output=self.fc(context)
        return self.norm(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*4, bias=False),
            nn.GELU(),
            nn.Linear(d_model*4, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(0.3)
    def forward(self,inputs):
        residual = inputs
        output = self.drop(self.fc(inputs))
        return self.norm(output + residual)
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs,attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self,dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self,enc_inputs, bert_enc_outputs):
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            bert_enc_outputs, enc_self_attn = layer(bert_enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return bert_enc_outputs, enc_self_attns
    
class Decoder(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self,dec_inputs, enc_inputs, enc_outputs, gpt_dec_outputs):
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            gpt_dec_outputs, dec_self_attn, dec_enc_attn = layer(gpt_dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return gpt_dec_outputs, dec_self_attns, dec_enc_attns

class GRA(nn.Module):
    def __init__(self, bert, gpt, vocabsize, d_model=768):
        super().__init__()
        self.bert = bert
        self.gpt = gpt
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocabsize, bias=False)
    def forward(self,enc_inputs,dec_inputs,masked_pos):
        enc_outputs_bert, _ = self.bert(enc_inputs,masked_pos)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs_bert)
        dec_outputs_gpt, _ = self.gpt(dec_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs,dec_outputs_gpt)
        dec_outputs =dec_outputs + dec_outputs_gpt
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

def create_parser():
    parser = argparse.ArgumentParser(description="GRA seq2seq",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=True)
    parser.add_argument("--tcr_vocab_path",dest="tcr_vocab_path",type=str,help="The vocab file in .csv format.",required=True)
    parser.add_argument("--pep_vocab_path",dest="pep_vocab_path",type=str,help="The vocab file in .csv format.",required=True)
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to save/load model.", required=True)
    parser.add_argument("--bert_path",dest="bert_path",type=str, help="the path to load bert.", required=True)
    parser.add_argument("--gpt_path",dest="gpt_path",type=str, help="the path to load gpt.", required=True)
    parser.add_argument("--result_path",dest="result_path",type=str, help="the path to store result.", required=False)
    parser.add_argument("--mode",dest="mode",type=str, help="train or generate", required=True)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=10, required=False)
    parser.add_argument("--beam",dest="beam",type=int,help="beam_size",default=500, required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=32, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=64, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=5e-5, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=0, help="seed for reproductbility",required=False)
    args = parser.parse_args()
    return args

def main(args):
    if args.mode == 'train':
        accelerator = Accelerator()
        set_seed(args.random_seed)
        dataset, validate_dataset, vocab_size_tcr, vocab_size_pep = make_data_for_seq2seq(args.data_path, args.tcr_vocab_path, args.pep_vocab_path, args.maxlen, args.mode)
        data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

        # device = accelerator.device
        bert = BERT(vocab_size_pep)
        gpt = GPT_Model(vocab_size_tcr)
        bert.load_state_dict(torch.load(args.bert_path))
        gpt.load_state_dict(torch.load(args.gpt_path))
        for name, param in bert.named_parameters():
            if name == 'linear.weight' or name == 'linear.bias':
                param.requires_grad = False     
        for param in gpt.parameters():
            param.requires_grad = False
        model = GRA(bert, gpt, vocab_size_tcr)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = AdamW([{"params": [p for p in model.parameters() if p.requires_grad]},],lr=args.learning_rate)
        total_steps = len(data_loader) * args.epoch
        warmup_steps = 0.1 * total_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
            model, optimizer, data_loader, validate_dataloader, scheduler
    )


        for epoch in range(args.epoch):
            model.train()
            loss = 0
            for dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos in data_loader:
                optimizer.zero_grad()
                logits_lm, _, _, _ = model(enc_input_ids, dec_input_ids, masked_pos)
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
                for dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos in validate_dataloader:
                    logits_lm, _, _, _ = model(enc_input_ids, dec_input_ids, masked_pos)
                    loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
                    loss_lm = loss_lm * mask_for_loss
                    loss_lm = (loss_lm.float()).mean()
                    loss_v = loss_v + loss_lm.item()
                accelerator.print(f"loss in validate_dataset:{loss_v}")

        
        accelerator.wait_for_everyone()
    
        model = accelerator.unwrap_model(model)
        state_dict = model.state_dict()
    
        accelerator.save(state_dict, args.model_path)
    if args.mode == 'generate':
        # accelerator = Accelerator()
        device = torch.device('cuda:0')
        dataset, tcr_idx2token, pep_idx2token, vocab_size_tcr, vocab_size_pep = make_data_for_seq2seq(args.data_path, args.tcr_vocab_path, args.pep_vocab_path, args.maxlen, args.mode)
        data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        bert = BERT(vocab_size_pep)
        gpt = GPT_Model(vocab_size_tcr)
        bert.load_state_dict(torch.load(args.bert_path))
        gpt.load_state_dict(torch.load(args.gpt_path))
        model = GRA(bert, gpt, vocab_size_tcr)
        model.load_state_dict(torch.load(args.model_path)) 
        model.to(device)  

        seq_list = []
        with torch.no_grad():
            for inputIds, mask in tqdm(data_loader):
                inputIds = inputIds.to(device)
                mask = mask.to(device)
                batchSize = inputIds.shape[0]
                inputEncodes,_ = model.bert(inputIds, mask)
                inputEncodes,_ = model.encoder(inputIds, inputEncodes)
                input_expanded_ids, input_expanded_encodes = expand_inputs(input_ids=inputIds, input_encodes=inputEncodes,
                                                                beam_size=args.beam)
                generated_beam_hyp_list = []
                for i in range(batchSize):
                    cur_input_id = inputIds[i, :]
                    generated_beam_hyp_list.append(
                        BeamHypotheses(num_beams=args.beam, max_length=32, length_penalty=0.7, input_ids=cur_input_id)
                    )
                decoder_output_ids = torch.full((batchSize * args.beam, 1), 1 ,dtype=torch.long).to(device)
                done = [False for _ in range(batchSize)]
                beam_scores = torch.zeros((batchSize*args.beam, 1), dtype=torch.float).to(device)
                cur_len = 1

                while cur_len < 32:
                    dec_outputs_gpt,_ = model.gpt(decoder_output_ids)
                    dec_outputs, _, _ = model.decoder(decoder_output_ids, input_expanded_ids, input_expanded_encodes, dec_outputs_gpt) 
                    dec_outputs =dec_outputs + dec_outputs_gpt
                    dec_logit = model.projection(dec_outputs)
                    dec_log_softmax_score = F.log_softmax(dec_logit, dim=-1)
                    next_token_log_softmax_score = dec_log_softmax_score[:,-1,:]
                    cur_dec_seq_log_softmax_score = next_token_log_softmax_score + beam_scores.expand_as(next_token_log_softmax_score)
                    cur_dec_seq_log_softmax_score = cur_dec_seq_log_softmax_score.view(batchSize, -1)
                    next_scores, next_tokens = torch.topk(cur_dec_seq_log_softmax_score, 2*args.beam, dim=1, largest=True, sorted=True)
                    next_batch_beam_list = []
                    for batch_idx in range(batchSize):
                        if done[batch_idx]:
                            next_batch_beam_list.extend([(0, 0, args.beam*batch_idx) for _ in range(args.beam)])
                            continue
                        cur_batch_beam_list = []
                        for bream_rank_id, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[batch_idx],next_scores[batch_idx])):
                            token_id = beam_token_id % vocab_size_tcr
                            beam_id = beam_token_id // vocab_size_tcr
                            effective_beam_id = batch_idx * args.beam + beam_id
                        
                            if token_id == 2:
                                generated_beam_hyp_list[batch_idx].add(hyp=decoder_output_ids[effective_beam_id].clone(), sum_log_probs=beam_token_score.item())
                            else:
                                cur_batch_beam_list.append((beam_token_score, token_id, effective_beam_id))

                            if len(cur_batch_beam_list) == args.beam:
                                break
                        
                        while len(cur_batch_beam_list) < args.beam:
                            cur_batch_beam_list.append((0, 0, args.beam*batch_idx))

                        next_batch_beam_list.extend(cur_batch_beam_list)
                        done[batch_idx] = done[batch_idx] or generated_beam_hyp_list[batch_idx].is_done(next_scores[batch_idx].max().item(), cur_len=cur_len)
                    if all(done):
                        break 
                    cur_beam_scores = [x[0] for x in next_batch_beam_list]
                    beam_scores = torch.tensor(cur_beam_scores).unsqueeze(1).to(device)

                    next_token_ids = [x[1] for x in next_batch_beam_list]
                    next_token_ids = torch.tensor(next_token_ids).unsqueeze(1).to(device)
                    effective_beam_ids = [x[2] for x in next_batch_beam_list]
                    decoder_output_ids = torch.cat([decoder_output_ids[effective_beam_ids,:],next_token_ids],dim=-1).to(device)
                    cur_len += 1    
                
                for batch_idx in range(batchSize):
                    if done[batch_idx]:
                        continue
                    for beam_id in range(args.beam):
                        effective_beam_id = batch_idx * args.beam + beam_id
                        final_score = beam_scores[effective_beam_id].item()
                        final_tokens = decoder_output_ids[effective_beam_id].clone()
                        generated_beam_hyp_list[batch_idx].add(final_tokens, final_score)

                # for i, hypotheses in enumerate(generated_beam_hyp_list):
                for batch_idx in range(batchSize):
                    sorted_hyps = sorted(generated_beam_hyp_list[batch_idx].beams, key=lambda x: x[0], reverse=True)
                    # best_hyps = sorted_hyps[0]
                    pep_seq = (generated_beam_hyp_list[batch_idx].input_ids).to('cpu').tolist()
                    pep = ''
                    for value in pep_seq:
                        if value == 0:
                            break
                        elif value == 1 or value == 2:
                            pass
                        else:
                            pep = pep + pep_idx2token[value]   
                    for i in range(args.beam):
                        tcr = ''
                        tcr_seq = sorted_hyps[i][1].to('cpu').tolist()
                        for value in tcr_seq:
                            if value == 1:
                                pass
                            else:
                                tcr = tcr + tcr_idx2token[value]
                        seq_list.append([pep,tcr])

            df = pd.DataFrame(seq_list, columns=['epitope', 'beta'])
            df.to_csv(args.result_path, index=False)

                            
if __name__=="__main__":
    args=create_parser()
    main(args)

