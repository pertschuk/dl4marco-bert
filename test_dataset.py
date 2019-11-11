# from run_msmarco import input_fn_builder
import tensorflow as tf
from server import feature_generator, input_q
import collections
import tensorflow_datasets as tfds
import numpy as np

MAX_EVAL_EXAMPLES = 100000

def add_to_q(dataset_path):
    queries_docs = collections.defaultdict(list)
    queries = {}
    doc_dict = {}
    num_eval_docs = 1000
    doc_dict["00000000"] = 'FAKE DOC'
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            queries_docs[query_id].append((doc_id, doc))
            doc_dict[doc_id] = doc
            queries[query_id] = query

    with open('query_doc_ids_dev.txt', 'r') as file:
        docs = []
        queries_list = set()
        for i, line in enumerate(file):
            query_id, doc_id = line.strip().split('\t')
            query = queries[query_id]
            docs.append(doc_dict[doc_id])
            queries_list.add(query)
            if (i+1) % 1000 == 0:
                input_q.put((query, docs))
                docs = []
                assert len(queries_list) == 1
                queries_list = set()
                print(i)
            if i > MAX_EVAL_EXAMPLES:
                break

def main():
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 1
    dataset_path = 'dataset_dev.tf'
    slice_dataset = input_fn_builder(dataset_path, MAX_SEQ_LENGTH, False,
                                     max_eval_examples=MAX_EVAL_EXAMPLES)
    og_dataset = slice_dataset(params={"batch_size": BATCH_SIZE})

    output_types = {
        "input_ids": tf.int32,
        "segment_ids": tf.int32,
        "input_mask": tf.int32,
    }
    dataset = tf.data.Dataset.from_generator(feature_generator, output_types)
    dataset = dataset.padded_batch(
        batch_size=BATCH_SIZE,
        padded_shapes={
            "input_ids": [MAX_SEQ_LENGTH],
            "segment_ids": [MAX_SEQ_LENGTH],
            "input_mask": [MAX_SEQ_LENGTH],
        },
        padding_values={
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0
        },
        drop_remainder=True)
    add_to_q('data/top1000.dev')
    print(og_dataset)
    print(dataset)
    for og_features, new_features in zip(
            tfds.as_numpy(og_dataset),tfds.as_numpy(dataset)):
        assert np.equal(og_features['input_ids'], new_features['input_ids']).all()
        assert np.equal(og_features['segment_ids'], new_features['segment_ids']).all()
        assert np.equal(og_features['input_mask'], new_features['input_mask']).all()

if __name__ == '__main__':
    main()