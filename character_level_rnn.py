# Imports for character level RNN
from character_rnn.encoder import Model
from matplotlib import pyplot as plt
import pandas as pd
from character_rnn.utils import sst_binary, train_with_reg_cv

### Make prediction for Character Level RNN ###
char_rnn_model = Model()

trX, vaX, teX, trY, vaY, teY = sst_binary()

trXt = char_rnn_model.transform(trX)
vaXt = char_rnn_model.transform(vaX)
teXt = char_rnn_model.transform(teX)

# classification results
full_rep_acc, c, nnotzero, char_rnn_model = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)

char_rnn_model_prediction = char_rnn_model.predict_proba(trXt)
char_rnn_model_prediction = [item[1] for item in char_rnn_model_prediction]
training_char_rnn_df = pd.DataFrame(list(zip(trX, trY, char_rnn_model_prediction)),
                                    columns=['sentence', 'sentiment', 'char_prediction'])

char_rnn_model_prediction = char_rnn_model.predict_proba(vaXt)
char_rnn_model_prediction = [item[1] for item in char_rnn_model_prediction]
dev_char_rnn_df = pd.DataFrame(list(zip(vaX, vaY, char_rnn_model_prediction)),
                               columns=['sentence', 'sentiment', 'char_prediction'])

char_rnn_model_prediction = char_rnn_model.predict_proba(teXt)
char_rnn_model_prediction = [item[1] for item in char_rnn_model_prediction]
testing_char_rnn_df = pd.DataFrame(list(zip(teX, teY, char_rnn_model_prediction)),
                                   columns=['sentence', 'sentiment', 'char_prediction'])

print('Performance of Character level RNN is:')
print('%05.2f test accuracy' % full_rep_acc)
print('%05.2f regularization coef' % c)
print('%05d features used' % nnotzero)

## visualize sentiment unit
#sentiment_unit = trXt[:, 2388]
#plt.hist(sentiment_unit[trY == 0], bins=25, alpha=0.5, label='neg')
#plt.hist(sentiment_unit[trY == 1], bins=25, alpha=0.5, label='pos')
#plt.legend()
#plt.show()

import os
data = pd.read_csv('character_rnn/data/imdb_ZSL_prediction_result.csv', encoding='cp1250')
data = data.dropna()
#data = data[:500]
print('Length of data after dropna is ', len(data))
data['sentiment'] = char_rnn_model_prediction
data.to_csv('imdb_ZSL_prediction_with_sentiment.csv')

training_char_rnn_df.to_pickle('training_char_rnn_df.pickle')
dev_char_rnn_df.to_pickle('dev_char_rnn_df.pickle')
#testing_char_rnn_df.to_pickle('testing_char_rnn_df.pickle')
testing_char_rnn_df.to_excel('ZSL_character_RNN_sentiement', engine='xlsxwriter')