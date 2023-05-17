import pandas as pd
import nltk
from nltk.util import ngrams

# Load the dataset into a pandas DataFrame
df = pd.read_csv('../train_data.csv')

# Define the function for generating n-grams
def generate_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

# Create a new column in the DataFrame to store the n-grams
n = 2 # Change this to generate different n-grams
df['ngrams'] = df['Text_data'].apply(lambda x: generate_ngrams(x, n))

# Save the DataFrame to a new CSV file
df.to_csv('dataset_with_ngrams.csv', index=False)
