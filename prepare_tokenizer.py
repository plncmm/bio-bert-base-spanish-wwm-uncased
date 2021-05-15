import transformers
import nltk
import re
import shutil
import argparse
import multiprocessing as mp
import collections
import itertools
import json

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

def tokenize(text):
  return nltk.word_tokenize(normalizer(text))

if __name__ == "__main__":
  
  nltk.download('punkt')

  tokenizer = transformers.BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", cache_dir = "/mnt/flock/fplana/cache")

  parser = argparse.ArgumentParser()

  parser.add_argument("--corpus")
  
  parser.add_argument("-p", type=int)

  args = parser.parse_args()
  
  lines = []
  with open(args.corpus, encoding="utf-8") as f:
    for line in f:
      lines.append(line.rstrip())
    
  p = args.p if args.p else mp.cpu_count()
  
  pool = mp.Pool(processes=p)

  tokens = pool.map(tokenize, lines)
  
  tokens = itertools.chain(*tokens)

  freqs = collections.Counter(tokens)

  vocab = [token for token,freq in freqs.items() if freq > 5]
  
  with open("vocab.json", "w", encoding="utf-8") as j:
    j.write(json.dumps(vocab, ensure_ascii=False))
  
  with open("freqs.json", "w", encoding="utf-8") as j:
    j.write(json.dumps(dict(freqs.most_common()), ensure_ascii=False))

  tokenizer.add_tokens(vocab)

  tokenizer.save_pretrained("models/bio-bert-spanish-tokenizer")

  shutil.copyfile("models/bio-bert-spanish-tokenizer/tokenizer_config.json", "models/bio-bert-spanish-tokenizer/config.json")