from run_msmarco import input_fn_builder
import tensorflow as tf
from server import feature_generator, input_q
import collections


def add_to_q(dataset_path):
    queries_docs = collections.defaultdict(list)
    query_ids = {}
    set_name = 'dev'
    num_eval_docs = 1000
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            queries_docs[query].append((doc_id, doc))
            query_ids[query] = query_id

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
    dataset_path = 'dataset_train.tf'
    slice_dataset = input_fn_builder(dataset_path,MAX_SEQ_LENGTH,False)
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
    for og_features, new_features in zip(og_dataset.make_one_shot_iterator().get_next(),
                                         dataset.make_one_shot_iterator().get_next()):
        print(og_features)
        print(new_features)
        assert og_features == new_features

if __name__ == '__main__':
    main()