import transformers
import nltk
nltk.download('punkt')
import re
import shutil
import argparse
import multiprocessing as mp

def normalizer(text, remove_tildes = False): #normalizes a given string to lowercase and changes all vowels to their base form
    text = text.lower() #string lowering
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text) #replaces every punctuation with a space
    if remove_tildes:
        text = re.sub('á', 'a', text) #replaces special vowels to their base forms
        text = re.sub('é', 'e', text)
        text = re.sub('í', 'i', text)
        text = re.sub('ó', 'o', text)
        text = re.sub('ú', 'u', text)
    return text

tokenizer = transformers.BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", cache_dir = "/mnt/flock/fplana/cache")

parser = argparse.ArgumentParser()

parser.add_argument("--corpus")

args = parser.parse_args()

with open(args.corpus, encoding="utf-8") as f:
  corpus = normalizer(f.read())
  
pool = mp.Pool(processes=4)

tokens = pool.map()

tokens = nltk.word_tokenize(corpus)

tokenizer.add_tokens(tokens)

tokenizer.save_pretrained("models/bio-bert-spanish-tokenizer")

shutil.copyfile("models/bio-bert-spanish-tokenizer/tokenizer_config.json", "models/bio-bert-spanish-tokenizer/config.json")