#!/bin/bash

python run_mlm.py --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased --tokenizer_name models/bio-bert-spanish-tokenizer --train_file data/raw/corpus_not_normalized.txt --do_train --line_by_line --output_dir models/bio-bert-base-spanish-wwm-uncased