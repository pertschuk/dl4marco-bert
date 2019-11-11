from run_msmarco import input_fn_builder
import tensorflow as tf
from server import feature_generator, input_q
import collections
import tensorflow_datasets as tfds
import numpy as np

MAX_EVAL_EXAMPLES = 10000

def add_to_q(dataset_path):
    queries_docs = collections.defaultdict(list)
    query_ids = {}
    num_eval_docs = 1000
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            queries_docs[query].append((doc_id, doc))
            query_ids[query] = query_id
            if i > MAX_EVAL_EXAMPLES: break

    # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    for query in queries:
        docs = queries_docs[query]
        docs += max(
            0, num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
        queries_docs[query] = docs

    assert len(
        set(len(docs) == num_eval_docs for docs in queries_docs.values())) == 1, (
        'Not all queries have {} docs'.format(num_eval_docs))

    for i, (query, doc_ids_docs) in enumerate(queries_docs.items()):
        doc_ids, docs = zip(*doc_ids_docs)
        input_q.put((query, docs))

def main():
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 1
    dataset_path = 'dataset_dev.tf'
    slice_dataset = input_fn_builder(dataset_path,MAX_SEQ_LENGTH,False,
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
    for og_features, new_features in zip(
            tfds.as_numpy(og_dataset),tfds.as_numpy(dataset)):
        assert np.equal(og_features['input_ids'].all(), new_features['input_ids'].all())
        assert np.equal(og_features['segment_ids'].all(), new_features['segment_ids'].all())
        assert np.equal(og_features['input_mask'].all(), new_features['input_mask'].all())

if __name__ == '__main__':
    main()