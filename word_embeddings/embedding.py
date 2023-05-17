import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the preprocessed data from CSV file
data = pd.read_csv('./preprocessed_data.csv')

# Convert the text to sequences of integers
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['Text_data'])
sequences = tokenizer.texts_to_sequences(data['Text_data'])

# Pad the sequences to ensure they have the same length
max_seq_len = 100
padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')

# Convert the labels to one-hot encoded vectors
labels = to_categorical(data['Label'])

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Save the preprocessed data to NumPy arrays
np.save('train_data.csv', train_data)
np.save('train_labels.csv', train_labels)
np.save('test_data.csv', test_data)
np.save('test_labels.csv', test_labels)
DF = pd.DataFrame()
 
# save the dataframe as a csv file
DF.to_csv("data1.csv")