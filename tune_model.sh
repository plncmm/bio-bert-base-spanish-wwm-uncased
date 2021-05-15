#!/bin/bash
#SBATCH -J bio-bert-spanish
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o bio-bert-spanish_%j.out
#SBATCH -e bio-bert-spanish_%j.err
#SBATCH --mail-user=correo@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --mem-per-cpu=4300
#SBATCH --gres=gpu:2

ml Python/3.7.3
ml CUDA/10.2.89

/home/fplana/fvillena/bio-bert-base-spanish-wwm-uncased/venv/bin/python run_mlm.py --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased --tokenizer_name models/bio-bert-spanish-tokenizer --train_file data/raw/corpus_not_normalized.txt --do_train --line_by_line --output_dir models/bio-bert-base-spanish-wwm-uncased --cache_dir /mnt/flock/fplana/cache