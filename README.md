# GRATCR

GRATCR is a pre-trained deep model capable of generating T cell receptors for the given epitopes.

## Requirements

GRATCR is trained on two GeForce RTX 3080 GPUs with 20G memory under the PyTorch 1.10.1 framework 

with Python 3.9. You can run the following commands to establish environment and install dependency packages.

```bash
conda create -n GRATCR
conda activate GRATCR
conda install -r requirements.txt
```

## Model Pre-training

To train the BERT, run:

```commandline
accelerate launch BERT.py --data_path='./Data/merged/epitope.csv' --vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/bert_pretrain.pth" 
```
To train the GPT, run:

```commandline
accelerate launch GPT.py --data_path='./Data/merged/beta.csv' --vocab_path='./Data/vocab/total-beta.csv' --model_path="./model/gpt_pretrain.pth"
```

## Model Fine-tuning

To fine-tune the GRATCR, run:

```commandline
accelerate launch GRA.py --data_path='./Data/MIRA/MIRA.csv' --tcr_vocab_path='./Data/vocab/total-beta.csv' --pep_vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/gra.pth" --bert_path="./model/bert_pretrain.pth" --gpt_path="./model/gpt_pretrain.pth" --mode='train'
```

## TCR Generation

To generate TCRs for the given epitopes, run:

```commandline
python GRA.py --data_path='./Data/MIRA/MIRA.csv' --tcr_vocab_path='./Data/vocab/total-beta.csv' --pep_vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/gra.pth" --bert_path="./model/bert_pretrain.pth" --gpt_path="./model/gpt_pretrain.pth" --mode='generate' --result_path='./result.csv'
```









