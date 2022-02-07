import json, bz2
import logging, pickle
import sys
import numpy as np

from bs4 import BeautifulSoup

import argparse
import collections
from typing import Dict, List, Tuple, Optional

from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def cosine(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.dot(a,b) / (norm_a * norm_b)

def get_Kbest(index, query_emb, K):
    distances = {}
    for (doc, emb) in index:
        score = cosine(emb, query_emb)
        try :
            max_score = distances[doc]
            if score > max_score:
                distances[doc]=score
        except KeyError:
            distances[doc]=score
        #distances.append((doc, score))
    res = sorted(distances.items(), key=lambda item: -item[1])
    #distances.sort(key = lambda t : -t[1])
    #return distances[:K]
    return res[:K]

def parse_args():
    parser = argparse.ArgumentParser(
        'Searching with DPR on the TechQA question dataset')
    parser.add_argument('index_file', metavar='index',
                        help='File where the DPR index is stored')
    parser.add_argument('q_file', metavar='questions.json',
                        help='Question file in JSON format')
    parser.add_argument('--out-file', '-o', metavar='pred.json',
                        help='Write predictions to file (default is stdout).')
    parser.add_argument('--verbose', '-v', action="store_const", const=logging.DEBUG,
                        default=logging.INFO)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

print("loading model...")
tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
print("done.")
"""
#for debug:
inf = bz2.BZ2File("../training_dev_technotes.json.bz2", 'r')
collection={}
docs = json.load(inf)
for k in docs.keys():
    data=docs[k]["content"]
    soup = BeautifulSoup(data, 'html.parser')
    text=soup.get_text()
    collection[k]=text
"""
def main(OPTS):
    logging.basicConfig(level=OPTS.verbose)

    out_pred={}
    predictions={}
    idx = open(OPTS.index_file, "rb")
    index = pickle.load(idx) # (id, embedding) list
    idx.close()

    with open(OPTS.q_file, encoding='utf-8') as f:
        dataset = {query['QUESTION_ID']: query for query in json.load(f)}
        for qid, q in dataset.items():
            txt=q['QUESTION_TITLE']
            answerable=q['ANSWERABLE']
            #if answerable == 'N': continue
            print(qid)
            predictions[qid]=list()

            input_ids = tokenizer(txt, return_tensors='pt')["input_ids"]
            query_embeddings = model(input_ids).pooler_output
            #print("Q: ", txt)
            results=get_Kbest(index, query_embeddings[0].detach().numpy(), K=5)
            """
            for r in results:
                print(r)
                print(collection[r[0]])
                print('------------------------------------')
            """
            pred={}
            if len(results) == 0 :
                #No answer Case (should never happen)
                pred["doc_id"]=""
                pred["score"]=0
                predictions[qid].append(pred)
            else:
                i=1
                for r in results:
                    #print(i, r[0], r[1])
                    pred["doc_id"]=r[0]
                    pred["score"]=str(r[1])
                    predictions[qid].append(pred)
                    i+=1

    out_pred["predictions"]=predictions
    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(out_pred, f)
    else:
        print(json.dumps(out_pred, indent=2))
    return out_pred

if __name__ == '__main__':
    OPTS = parse_args()
    main(OPTS)
