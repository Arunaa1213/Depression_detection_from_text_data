import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# Load the preprocessed data from CSV file
data = pd.read_csv('preprocessed_data.csv')

# Split the preprocessed text data into individual sentences
sentences = [text.split() for text in data['Text_data']]

# Train a Word2Vec model on the sentences
model = Word2Vec(sentences,window=5, min_count=1, workers=4)

# Create an empty matrix to store the embeddings
embeddings = np.zeros((len(sentences), 100))

# Fill in the matrix with the embeddings for each sentence
for i, sentence in enumerate(sentences):
    embeddings[i] = np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0)

# Concatenate the embeddings with the PID and Label columns
preprocessed_data = pd.concat([data[['PID', 'Label']], pd.DataFrame(embeddings)], axis=1)

# Save the preprocessed data to CSV file
preprocessed_data.to_csv('word2vec_data.csv', index=False)
