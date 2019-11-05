from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

training_char_rnn_df = pd.read_pickle('training_char_rnn_df.pickle')
dev_char_rnn_df = pd.read_pickle('dev_char_rnn_df.pickle')
testing_char_rnn_df = pd.read_pickle('testing_char_rnn_df.pickle')

training_word_rnn_df = pd.read_pickle('training_word_rnn_df.pickle')
dev_word_rnn_df = pd.read_pickle('dev_word_rnn_df.pickle')
testing_word_rnn_df = pd.read_pickle('testing_word_rnn_df.pickle')