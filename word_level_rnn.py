# Imports for Word Level RNN
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import time
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import pandas as pd
import word_level_rnn.graphs

### Make prediction for Word Level RNN with adverserial training ###

# Flags governing adversarial training are defined in adversarial_losses.py.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', 'word_level_rnn/tmp/models/imdb_classify',
                    'Directory where to read model checkpoints.')
flags.DEFINE_string('eval_dir', 'word_level_rnn/tmp/models/imdb_eval',
                    'Directory where to write event logs.')
flags.DEFINE_string('eval_data', 'test', 'Specify which dataset is used. '
                                         '("train", "valid", "test") ')
flags.DEFINE_bool('run_once', True, 'Whether to run eval only once.')
flags.DEFINE_integer('num_examples', 10240, 'Number of examples to run.')
flags.DEFINE_string('master', '',
                    'BNS name prefix of the Tensorflow eval master, '
                    'or "local".')
flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run the eval.')


def restore_from_checkpoint(sess, saver):
    """Restore model from checkpoint.

  Args:
    sess: Session.
    saver: Saver for restoring the checkpoint.

  Returns:
    bool: Whether the checkpoint was found and restored
  """
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        tf.logging.info('No checkpoint found at %s', FLAGS.checkpoint_dir)
        return False

    saver.restore(sess, ckpt.model_checkpoint_path)
    return True


def run_prediction(prediction_ops, saver):
    """Runs evaluation over FLAGS.num_examples examples.

  Args:
    prediction_ops: dict<prediction name, value>
    saver: Saver.

  Returns:
    value
  """
    sv = tf.train.Supervisor(
        logdir=FLAGS.eval_dir, saver=None, summary_op=None, summary_writer=None)

    with sv.managed_session(
            master=FLAGS.master, start_standard_services=False) as sess:
        if not restore_from_checkpoint(sess, saver):
            return
        sv.start_queue_runners(sess)
        num_batches = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        print('Running', num_batches, 'batches for prediction.')
        prediction_ary = np.array([])
        label_ary = np.array([])
        index_ary = np.array([])
        sentence_ary = []
        for i in range(num_batches):
            print('Running batch', i + 1, '/', num_batches)

            ops = (prediction_ops['prediction'], prediction_ops['label'],
                   prediction_ops['index'], prediction_ops['input_sentence'])
            prediction, labels, indices, input_sentence = sess.run(ops)
            prediction_ary = np.append(prediction_ary, prediction.flatten())
            label_ary = np.append(label_ary, labels.flatten())
            index_ary = np.append(index_ary, indices.flatten())

            for sentence in input_sentence:
                sentence_ary.append(sentence.decode("utf-8"))

        return prediction_ary, label_ary, index_ary, sentence_ary


def main(_):
    print('starting')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    tf.logging.info('Building prediction graph...')

    dataset = 'test'
    if dataset == 'train':
        output = word_level_rnn.graphs.get_model().prediction_graph(dataset='train')
        prediction_ops, moving_averaged_variables = output
        saver = tf.train.Saver(moving_averaged_variables)
        prediction_ary, label_ary, index_ary, sentence_ary = run_prediction(prediction_ops, saver)
        training_word_rnn_df = pd.DataFrame(list(zip(sentence_ary, label_ary, prediction_ary)),
                                            columns=['sentence', 'sentiment', 'prediction'])
        print(training_word_rnn_df)
        training_word_rnn_df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
        print(training_word_rnn_df)
        training_word_rnn_df.to_pickle('training_word_rnn_df.pickle')

    elif dataset == 'valid':
        output = word_level_rnn.graphs.get_model().prediction_graph(dataset='valid')
        prediction_ops, moving_averaged_variables = output
        saver = tf.train.Saver(moving_averaged_variables)
        prediction_ary, label_ary, index_ary, sentence_ary = run_prediction(prediction_ops, saver)
        dev_word_rnn_df = pd.DataFrame(list(zip(sentence_ary, label_ary, prediction_ary)),
                                       columns=['sentence', 'sentiment', 'prediction'])
        print(dev_word_rnn_df)
        dev_word_rnn_df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
        print(dev_word_rnn_df)
        dev_word_rnn_df.to_pickle('dev_word_rnn_df.pickle')

    elif dataset == 'test':
        output = word_level_rnn.graphs.get_model().prediction_graph(dataset='test')
        prediction_ops, moving_averaged_variables = output
        saver = tf.train.Saver(moving_averaged_variables)
        prediction_ary, label_ary, index_ary, sentence_ary = run_prediction(prediction_ops, saver)
        testing_word_rnn_df = pd.DataFrame(list(zip(sentence_ary, label_ary, prediction_ary)),
                                           columns=['sentence', 'sentiment', 'prediction'])
        print(testing_word_rnn_df)
        testing_word_rnn_df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
        print(testing_word_rnn_df)
        testing_word_rnn_df.to_pickle('testing_word_rnn_df.pickle')

    # print(accuracy_score(label_ary, prediction_ary))


if __name__ == '__main__':
    tf.app.run()
