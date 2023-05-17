import pandas as pd
import re
import string
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')

# Load the test dataset
test_df = pd.read_csv('test_data.csv')

# Remove special characters and punctuations
def remove_special_chars(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text) # remove punctuations
        text = re.sub(r'[^\x00-\x7f]', '', text) # remove non-ascii characters
        text = re.sub(r'\d+', '', text) # remove digits
        text = text.lower() # convert to lowercase
        return text
    else:
        return ""

test_df['Text data'] = test_df['Text data'].apply(lambda x: remove_special_chars(x))

# Convert to lowercase
test_df['Text data'] = test_df['Text data'].str.lower()

# Tokenize
test_df['Text data'] = test_df['Text data'].apply(lambda x: word_tokenize(x))

# Remove stop words
stop_words = set(stopwords.words('english'))
test_df['Text data'] = test_df['Text data'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert list of words to string
test_df['Text data'] = test_df['Text data'].apply(lambda x: ' '.join(x))

# Save the preprocessed test dataset
test_df.to_csv('preprocessed_test_data.csv', index=False)
