import json
import logging
import sys

import argparse
import collections
from typing import Dict, List, Tuple, Optional

_NEGATIVE_INFINITY = float('-inf')


class EVAL_OPTS():
    def __init__(self, data_file, pred_file, out_file="", top_k=5,
                 out_image_dir=None, verbose=False):
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_file = out_file
        self.verbose = verbose
        self.top_k = top_k


OPTS = EVAL_OPTS(data_file=None, pred_file=None)


def parse_args():
    parser = argparse.ArgumentParser(
        'Unofficial evaluation script for TechQA v0.2. It will produce the following metrics: '
        '\n"MRR": Mean Reciprocal Rank')
    parser.add_argument('data_file', metavar='dev_vX.json',
                        help='Input competition query annotations JSON file.')
    parser.add_argument('pred_file', metavar='pred.json',
                        help='Model predictions - plain text file with the format: '
                             '{                               '
                             '   "predictions": {             '
                             '     "QID1": [                  '
                             '       {                        '
                             '         "doc_id": "swg234",    '
                             '         "score": 3.4      '
                             '       },                       '
                             '       {                        '
                             '         "doc_id": "swg234",    '
                             '         "score": 3      '
                             '       }...                        '
                             '     ],                         '
                             '     "QID2": [                  '
                             '       {                        '
                             '         "doc_id": "",          '
                             '         "score": 0       '
                             '       },                       '
                             '       {                        '
                             '         "doc_id": "swg123",    '
                             '         "score": -1       '
                             '       }...                        '
                             '     ]...                          '
                             '   }                            '
                             '}')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--top_k', '-k', type=int, default=10,
                        help='Eval script will compute F1 score using the top 1 prediction'
                             ' as well as the top k predictions')
    parser.add_argument('--verbose', '-v', action="store_const", const=logging.DEBUG,
                        default=logging.INFO)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for qid, q in dataset.items():
        if 'ANSWERABLE' in q and q['ANSWERABLE'] == 'Y':
            qid_to_has_ans[qid] = True
        else:
            qid_to_has_ans[qid] = False

    return qid_to_has_ans


def get_raw_scores(
        dataset: Dict[str, Dict], preds: Dict[str, List[Dict]],
        qid_to_has_ans: Dict[str, bool], top_k: int) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]]]:
    mrr_scores_by_qid = {}

    for qid, q in dataset.items():

        if qid not in preds or len(preds[qid]) < 1:
            logging.warning('Missing predictions for %s; going to receive 0 points for it' % qid)
            # Force this score to be incorrect
            mrr_scores.append(0)
        else:
            if qid_to_has_ans[qid]:
                gold_doc_id = q['DOCUMENT']
            else:
                gold_doc_id = ''

            mrr=0.0
            i=1.0
            for prediction in preds[qid][:top_k]:
                pred_doc=prediction['doc_id']
                pred_score=prediction['score']
                if (pred_doc == gold_doc_id) and (gold_doc_id != ''):
                    print(pred_doc, gold_doc_id)
                    mrr=1.0/i
                    break
                i+=1.0
        print(qid, mrr)
        mrr_scores_by_qid[qid] = mrr
    return mrr_scores_by_qid


def make_eval_dict(mrr_scores_by_qid: Dict[str, float],
                   qid_list: Optional[set] = None) -> collections.OrderedDict:
    mrr_score_sum = 0

    if not qid_list:
        total = len(mrr_scores_by_qid)
        for scores in mrr_scores_by_qid.values():
            mrr_score_sum += scores
    else:
        total = len(qid_list)
        for qid in qid_list:
            mrr_score_sum += mrr_scores_by_qid[qid]
    print("mrr score sum:", mrr_score_sum)
    return collections.OrderedDict([
        ('mrr', (mrr_score_sum / total)),
        ('total', total),
    ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def main(OPTS):
    logging.basicConfig(level=OPTS.verbose)

    with open(OPTS.data_file, encoding='utf-8') as f:
        dataset = {query['QUESTION_ID']: query for query in json.load(f)}
    with open(OPTS.pred_file, encoding='utf-8') as f:
        system_output = json.load(f)
        preds = system_output['predictions']

    out_eval = evaluate(preds=preds, dataset=dataset)

    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))
    return out_eval


def evaluate(preds: Dict[str, List[Dict]], dataset: Dict[str, Dict]):
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = {k for k, v in qid_to_has_ans.items() if v}
    no_ans_qids = {k for k, v in qid_to_has_ans.items() if not v}

    # Calculate metrics without thresholding
    mrr_score_by_qid = get_raw_scores(dataset, preds, qid_to_has_ans, OPTS.top_k)
    top1_pred_score_by_qid = {qid: scores for qid, scores in mrr_score_by_qid.items()}

    # Create evaluation summary
    out_eval = make_eval_dict(mrr_score_by_qid)
    if has_ans_qids:
        merge_eval(out_eval, make_eval_dict(mrr_score_by_qid, qid_list=has_ans_qids), 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(mrr_score_by_qid, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')

    return out_eval

if __name__ == '__main__':
    OPTS = parse_args()
    main(OPTS)
