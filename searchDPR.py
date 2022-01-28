import json
import logging, pickle
import sys

import argparse
import collections
from typing import Dict, List, Tuple, Optional



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

def main(OPTS):
    logging.basicConfig(level=OPTS.verbose)

    out_pred={}
    predictions={}
    index = pickle.load(OPTS.index)
    with open(OPTS.q_file, encoding='utf-8') as f:
        dataset = {query['QUESTION_ID']: query for query in json.load(f)}


        for qid, q in dataset.items():
            txt=q['QUESTION_TITLE']
            answerable=q['ANSWERABLE']
            #if answerable == 'N': continue
            print(qid)
            predictions[qid]=list()
            with ix.searcher() as searcher:
                query = QueryParser("body", ix.schema).parse(txt)
                results = searcher.search(query)
                pred={}
                if len(results) == 0 :
                    #No answer Case
                    pred["doc_id"]=""
                    pred["score"]=0
                    predictions[qid].append(pred)
                else:
                    i=1
                    for r in results:
                        #print(i, r['doc_id'], r.score)
                        pred["doc_id"]=r['doc_id']
                        pred["score"]=r.score
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
