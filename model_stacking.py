import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model_stacking(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l2',
                         C=2 ** np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed + i, solver='lbfgs')
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed + len(C), solver='lbfgs')
    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY) * 100.
    else:
        score = model.score(vaX, vaY) * 100.
    return score, c, nnotzero, model


training_char_rnn_df = pd.read_pickle('training_char_rnn_df.pickle')
dev_char_rnn_df = pd.read_pickle('dev_char_rnn_df.pickle')
testing_char_rnn_df = pd.read_pickle('testing_char_rnn_df.pickle')

training_word_rnn_df = pd.read_pickle('training_word_rnn_df.pickle')
dev_word_rnn_df = pd.read_pickle('dev_word_rnn_df.pickle')
testing_word_rnn_df = pd.read_pickle('testing_word_rnn_df.pickle')

# print(testing_char_rnn_df)
# print(testing_word_rnn_df)
training_df = pd.merge(training_char_rnn_df, training_word_rnn_df, how='inner', on=['sentence', 'sentiment'])
dev_df = pd.merge(dev_char_rnn_df, dev_word_rnn_df, how='inner', on=['sentence', 'sentiment'])
test_df = pd.merge(testing_char_rnn_df, testing_word_rnn_df, how='inner', on=['sentence', 'sentiment'])

# print(training_df)
# print(dev_df)
# print(test_df)

full_rep_acc, c, nnotzero, stacked_model = train_model_stacking(
    training_df[['prediction', 'char_prediction']].to_numpy(), training_df['sentiment'].to_numpy(),
    dev_df[['prediction', 'char_prediction']].to_numpy(), dev_df['sentiment'].to_numpy(),
    test_df[['prediction', 'char_prediction']].to_numpy(), test_df['sentiment'].to_numpy())
print('Performance of stacked model is:')
print('%05.2f test accuracy' % full_rep_acc)
print('%05.2f regularization coef' % c)
print('%05d features used' % nnotzero)
