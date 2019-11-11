"""Code to train and eval a BERT passage re-ranker on the MS MARCO dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from server import feature_generator, input_q
from threading import Thread
from queue import Queue
import collections
import csv

import os
import time
import numpy as np
import tensorflow as tf
import modeling

SEQ_LENGTH = 512

output_q = Queue()

bert_config_file = 'bert_marco/bert_config.json'
init_ckpt = 'bert_marco/bert_model.ckpt'
model_dir = 'bert_marco/'
batch_size = 4
num_eval_docs = 1000

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

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

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, log_probs) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={
        "log_probs": log_probs,
        "label_ids": label_ids,
      })
    return output_spec

  return model_fn


def input_fn(params):
  """The actual input function."""

  batch_size = params["batch_size"]

  output_types = {
      "input_ids": tf.int32,
      "segment_ids": tf.int32,
      "input_mask": tf.int32,
      "label_ids": tf.int32,
  }
  dataset = tf.data.Dataset.from_generator(feature_generator, output_types)

  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes={
          "input_ids": [SEQ_LENGTH],
          "segment_ids": [SEQ_LENGTH],
          "input_mask": [SEQ_LENGTH],
          "label_ids": [],
      },
      padding_values={
          "input_ids": 0,
          "segment_ids": 0,
          "input_mask": 0,
          "label_ids": 0,
      },
      drop_remainder=True)

  return dataset


def main(_):

  bert_config = modeling.BertConfig.from_json_file(bert_config_file)
  assert SEQ_LENGTH <= bert_config.max_position_embeddings

  run_config = tf.estimator.RunConfig(
      model_dir=model_dir)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=init_ckpt)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size)

  result = estimator.predict(input_fn=input_fn,
                             yield_single_examples=True)
  start_time = time.time()
  examples = 0
  results = []
  for item in result:
    results.append((item["log_probs"], item["label_ids"]))

    if len(results) == num_eval_docs:

      log_probs, labels = zip(*results)
      log_probs = np.stack(log_probs).reshape(-1, 2)

      scores = log_probs[:, 1]
      pred_docs = scores.argsort()[::-1]

      for doc_idx in pred_docs:
        output_q.put(doc_idx)
      examples += 1
      results = []
      print("took %s seconds" % ((time.time() - start_time)/ examples))


def pad_docs(docs, eval_size):
    if len(docs) < eval_size:
        return docs + [(0, 'FAKE DOC')] * (eval_size - len(docs))
    return docs


def output_fn():
    eval_size = 1000
    qrels = set()

    with open(os.path.join('data', 'qrels.dev.small.tsv')) as fh:
        data = csv.reader(fh, delimiter='\t')
        for qid, _, doc_id, _ in data:
            qrels.add((qid, doc_id))

    qid_count = collections.defaultdict(int)
    query_dict = dict()
    docs_dict = collections.defaultdict(list)
    with open('data/top1000.dev', 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            docs_dict[query_id].append((doc_id, doc))
            query_dict[query_id] = query

    for qid, query in query_dict.items():
        docs = pad_docs(docs_dict[qid], eval_size)
        doc_ids, doc_texts = zip(*docs)
        input_q.put((query, doc_texts))

        for rank in range(eval_size):
            doc_idx = output_q.get()
            doc_id = doc_ids[doc_idx]
            if (qid, doc_id) in qrels:
                print(qid, ' ', rank)
                qid_count[qid] = max(qid_count[qid], (1.0 / (float(rank+1))))
                mrr = sum(qid_count.values()) / len(qid_count.keys())
                print("MRR: %s " % mrr)


if __name__ == "__main__":
  t = Thread(target=output_fn)
  t.start()
  tf.app.run()
