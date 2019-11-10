import tensorflow as tf
import tokenization
import optimization
import modeling
import queue
import time
import numpy as np

MAX_SEQ_LENGTH = 64
num_labels = 2
VOCAB_FILE = 'bert_marco/vocab.txt'
bert_config_file = 'bert_marco/bert_config.json'
init_checkpoint = 'bert_marco/bert_model.ckpt'

q = queue.Queue()

tokenizer = tokenization.FullTokenizer(
      vocab_file=VOCAB_FILE, do_lower_case=True)

run_config = tf.estimator.RunConfig()


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, log_probs) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    initialized_variable_names = []
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log_probs": log_probs,
              "label_ids": label_ids,
          },
          scaffold_fn=scaffold_fn)

    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def feature_generator():
    while True:
        query, candidates = q.get()
        query = tokenization.convert_to_unicode(query)
        query_token_ids = tokenization.convert_to_bert_input(
            text=query, max_seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer,
            add_cls=True)

        query_token_ids_tf = tf.train.Feature(
            int64_list=tf.train.Int64List(value=query_token_ids))

        for i, doc_text in enumerate(candidates):
            doc_token_id = tokenization.convert_to_bert_input(
                text=tokenization.convert_to_unicode(doc_text),
                max_seq_length=MAX_SEQ_LENGTH - len(query_token_ids),
                tokenizer=tokenizer,
                add_cls=False)

            doc_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=doc_token_id))

            query_ids = tf.cast(query_token_ids_tf, tf.int32)
            doc_ids = tf.cast(doc_ids_tf, tf.int32)
            input_ids = tf.concat((query_ids, doc_ids), 0)

            query_segment_id = tf.zeros_like(query_ids)
            doc_segment_id = tf.ones_like(doc_ids)
            segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features = {
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
            }
            yield features


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
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

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def rank(query, candidates):
    q.put((query, candidates))
    batch_size = 16
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

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "log_probs": log_probs,
                "label_ids": label_ids,
            },
            scaffold_fn=scaffold_fn)

        return output_spec

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)


    result = estimator.predict(input_fn=input_fn,
                               yield_single_examples=True)
    start_time = time.time()
    results = []
    example_idx = 0
    total_count = 0
    for item in result:
        results.append((item["log_probs"], item["label_ids"]))
        tf.logging.info("Read {} examples in {} secs".format(
            total_count, int(time.time() - start_time)))

        log_probs, labels = zip(*results)
        log_probs = np.stack(log_probs).reshape(-1, 2)
        labels = np.stack(labels)

        scores = log_probs[:, 1]
        pred_docs = scores.argsort()[::-1]

if __name__ == '__main__':
    res = rank('This is a test', ['Test candidate'] * 100)