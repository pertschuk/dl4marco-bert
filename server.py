import tensorflow as tf
import tokenization
import optimization
import modeling
import queue
import time
import numpy as np
import csv
from threading import Thread
from collections import defaultdict

MAX_SEQ_LENGTH = 512
num_labels = 2
BATCH_SIZE = 4
VOCAB_FILE = 'bert_marco/vocab.txt'
bert_config_file = 'bert_marco/bert_config.json'
init_checkpoint = 'bert_marco/bert_model.ckpt'

input_q = queue.Queue()
output_q = queue.Queue()

tokenizer = tokenization.FullTokenizer(
      vocab_file=VOCAB_FILE, do_lower_case=True)

run_config = tf.estimator.RunConfig()



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    return (None, None, log_probs)


def rank():
    batch_size = BATCH_SIZE
    def input_fn():
        output_types = {
                "input_ids": tf.int32,
                "segment_ids": tf.int32,
                "input_mask": tf.int32,
            }
        dataset = tf.data.Dataset.from_generator(feature_generator, output_types)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
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
        return dataset

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            num_labels, False)

        tvars = tf.trainable_variables()

        initialized_variable_names = []
        print('****TRYING TO LOAD FROM INIT CHECKPOINT %s****' % init_checkpoint)
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print(" name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                "log_probs": log_probs
            })

        return output_spec

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    result = estimator.predict(input_fn=input_fn,
                               yield_single_examples=True)
    total = 0
    for item in result:
        tf.logging.info(result)
        total += 1
        output_q.put(item["log_probs"])


if __name__ == '__main__':
    data_dir ='data/'
    qrels = set()
    rank_thread = Thread(target=rank)
    rank_thread.start()
    with open(data_dir + 'qrels.dev.small.tsv') as fn:
        reader = csv.reader(fn, delimiter='\t')
        for qid, _, cid, _ in reader:
            qrels.add((qid, cid))

    dev_set = defaultdict(list)
    dev_queries = dict()
    dev_labels = defaultdict(list)

    with open(data_dir + 'top1000.dev') as fn:
        reader = csv.reader(fn, delimiter='\t')
        i = 0
        for qid, cid, query, passage in reader:
            dev_set[qid].append(passage)
            dev_queries[qid] = query
            dev_labels[qid].append(1 if (qid, cid) in qrels else 0)
            i += 1
            if i % 10000 == 0:
                print(i)

    # input_q.put(('test query', ['test cnadidate']*1000))

    total_mrr = 0
    start = time.time()
    for i, qid in enumerate(dev_set.keys()):
        query = dev_queries[qid]
        candidates = dev_set[qid]
        true_size = len(candidates)
        size = len(candidates)
        print('size %s' % size)

        padding = (BATCH_SIZE - (size % BATCH_SIZE)) % BATCH_SIZE
        candidates += [''] * padding
        size += padding

        assert len(candidates) % BATCH_SIZE == 0
        input_q.put((query, candidates))
        results = [output_q.get() for _ in range(size)][:true_size]
        log_probs = list(zip(*results))

        assert len(log_probs[0]) == size - padding
        assert len(log_probs[0]) == len(log_probs[1])
        log_probs = np.stack(log_probs).reshape(-1, 2)

        scores = log_probs[:, 1]
        pred_docs = scores.argsort()[::-1]
        rank = sum(np.arange(1, true_size + 1, dtype=float) * np.array(dev_labels[qid])[pred_docs])
        print(rank)
        relevant = np.array(dev_labels[qid])[pred_docs] * np.reciprocal(np.arange(1, true_size + 1, dtype=float))
        total_mrr += sum(relevant)
        print('Avg MRR: %s' % (total_mrr / (i+1)), 'Avg time %s' % ((time.time() - start)/(i+1)))

