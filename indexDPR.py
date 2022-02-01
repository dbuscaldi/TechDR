import json, bz2
import os, sys, os.path
from bs4 import BeautifulSoup
import pickle

import tracemalloc

from transformers import AutoTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

file_path = sys.argv[1]
inf = bz2.BZ2File(file_path, 'r') if file_path.endswith('.bz2') else  open(file_path, 'r')

print("loading pre-trained model...")
tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
print("done.")

def get_chunks(text, size):
    chunks = []
    c = text[:size]
    chunks.append(c)
    if len(text) > size:
        chunks += get_chunks(text[size:], size)
    return chunks

index = [] #list of pairs (key, embeddings)
i=0
docs = json.load(inf)
for k in docs.keys():
    #print(docs[k])
    data=docs[k]["content"]
    soup = BeautifulSoup(data, 'html.parser')
    text=soup.get_text()
    clean_text=os.linesep.join([s for s in text.splitlines() if s])
    #c = clean_text[:512] #truncate at 512 characters
    #chunks = [clean_text[i:i+512] for i in range(0, len(clean_text), 512)] #DPR can deal only with sentences of 512 characters max
    chunks = get_chunks(clean_text, 512)
    for c in chunks: #we'll store more vectors for the same document, if there are more chunks
        input_ids = tokenizer(c, return_tensors='pt')["input_ids"]
        embeddings = model(input_ids).pooler_output
        emb=embeddings[0].detach().numpy()
        #print(emb)
        index.append((k, emb))
    if i % 10 == 0:
        sys.stderr.write('.')
    if i % 800 == 0 and i > 0:
        sys.stderr.write('\n')
    sys.stderr.flush()
    i+=1


pickle.dump(index, open("DPRindex.pkl", "wb"), protocol=3)
