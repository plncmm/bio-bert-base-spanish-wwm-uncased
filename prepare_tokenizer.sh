#!/bin/bash
#SBATCH -J tokenizer-prepare
#SBATCH -p slims
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -o tokenizer-prepare_%j.out
#SBATCH -e tokenizer-prepare_%j.err
#SBATCH --mail-user=villenafabian@gmail.com
#SBATCH --mail-type=END,FAIL

ml Python/3.7.3

/home/fplana/fvillena/bio-bert-base-spanish-wwm-uncased/venv/bin/python prepare_tokenizer.py --corpus data/raw/corpus_not_normalized.txt -p 20