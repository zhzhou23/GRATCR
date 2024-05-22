# GRATCR

GRATCR is a pre-trained deep learning model designed to generate T cell receptors (TCRs) for specified epitopes.

## Requirements

GRATCR was trained on two GeForce RTX 3080 GPUs with 20GB memory using the PyTorch 1.10.1 framework and Python 3.9. To set up the environment and install the necessary dependencies, run the following commands:

```bash
conda create -n GRATCR
conda activate GRATCR
conda install -r requirements.txt
```

## Model Pre-training

To pre-train the BERT, run:

```commandline
accelerate launch BERT.py --data_path='./Data/merged/epitope.csv' --vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/bert_pretrain.pth" 
```
To pre-train the GPT, run:

```commandline
accelerate launch GPT.py --data_path='./Data/merged/beta.csv' --vocab_path='./Data/vocab/total-beta.csv' --model_path="./model/gpt_pretrain.pth"
```

## Model Fine-tuning

To fine-tune the GRATCR, run:

```commandline
accelerate launch GRA.py --data_path='./Data/MIRA/MIRA.csv' --tcr_vocab_path='./Data/vocab/total-beta.csv' --pep_vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/gra.pth" --bert_path="./model/bert_pretrain.pth" --gpt_path="./model/gpt_pretrain.pth" --mode='train'
```

## TCR Generation

To generate TCRs for given epitopes, run:

```commandline
python GRA.py --data_path='./Data/MIRA/MIRA.csv' --tcr_vocab_path='./Data/vocab/total-beta.csv' --pep_vocab_path='./Data/vocab/total-epitope.csv' --model_path="./model/gra.pth" --bert_path="./model/bert_pretrain.pth" --gpt_path="./model/gpt_pretrain.pth" --mode='generate' --result_path='./result.csv'
```









