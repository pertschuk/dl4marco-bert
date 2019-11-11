from collections import defaultdict
import csv
import os

DATA_PATH = 'data'

def main():
    qrels = set()
    hits = 0
    total = 0
    mrr = 0
    qid_count = defaultdict(int)

    with open(os.path.join(DATA_PATH, 'qrels.dev.small.tsv')) as fh:
        data = csv.reader(fh, delimiter='\t')
        for qid, _, doc_id, _ in data:
            qrels.add((qid, doc_id))
            qid_count[qid] += 1

    with open(os.path.join(DATA_PATH, 'output', 'msmarco_predictions_dev.tsv')) as fh:
        data = csv.reader(fh, delimiter='\t')
        last_qid = 0
        for qid, doc_id, rank in data:
            if (qid, doc_id) in qrels:
                qid_count[qid] = max(qid_count[qid], (1 / int(rank)))
            if qid != last_qid:
                mrr = sum(qid_count.values())
                total = len(qid_count.keys())
                print("MRR: %s " % (mrr / total))
            last_qid = qid

if __name__== '__main__':
    main()